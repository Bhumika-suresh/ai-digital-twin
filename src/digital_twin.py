"""
Digital Twin Simulation Module using SimPy.
Simulates a discrete-event textile manufacturing process:
Raw Material -> Warping -> Sizing -> Weaving -> Quality Inspection -> Finished Product.
Calculates stage utilizations, waiting times, completed/rejected yardage, and bottleneck data.
"""

import random
import simpy
import numpy as np
import pandas as pd

class TextileFactoryDigitalTwin:
    def __init__(
        self,
        sim_duration_hours=48.0,
        batch_size_yds=500.0,
        arrival_interval_hours=1.0,
        warping_machines=2,
        warping_time_hours=1.2,
        sizing_machines=2,
        sizing_time_hours=1.5,
        weaving_machines=6,
        weaving_time_hours=4.0,
        inspection_machines=2,
        inspection_time_hours=0.8,
        defect_prob=0.08,
        random_seed=42,
    ):
        self.sim_duration_hours = sim_duration_hours
        self.batch_size_yds = batch_size_yds
        self.arrival_interval_hours = arrival_interval_hours
        
        # Machine counts
        self.warping_machines = warping_machines
        self.sizing_machines = sizing_machines
        self.weaving_machines = weaving_machines
        self.inspection_machines = inspection_machines

        # Processing times
        self.warping_time_hours = warping_time_hours
        self.sizing_time_hours = sizing_time_hours
        self.weaving_time_hours = weaving_time_hours
        self.inspection_time_hours = inspection_time_hours

        # Quality parameters
        self.defect_prob = defect_prob
        self.random_seed = random_seed

        # Tracking state
        self.batches_started = 0
        self.batches_completed = 0
        self.batches_rejected = 0
        self.total_produced_yds = 0.0
        self.completed_yds = 0.0
        self.rejected_yds = 0.0

        # Stage metrics tracking
        self.stage_data = {
            "Warping": {"busy_time": 0.0, "wait_times": [], "queue_history": [], "processed_count": 0},
            "Sizing": {"busy_time": 0.0, "wait_times": [], "queue_history": [], "processed_count": 0},
            "Weaving": {"busy_time": 0.0, "wait_times": [], "queue_history": [], "processed_count": 0},
            "Inspection": {"busy_time": 0.0, "wait_times": [], "queue_history": [], "processed_count": 0},
        }
        self.event_log = []

    def _batch_process(self, env, batch_id, resources):
        """Simulates the lifecycle of a single textile production batch."""
        self.batches_started += 1
        stages = [
            ("Warping", resources["Warping"], self.warping_time_hours),
            ("Sizing", resources["Sizing"], self.sizing_time_hours),
            ("Weaving", resources["Weaving"], self.weaving_time_hours),
            ("Inspection", resources["Inspection"], self.inspection_time_hours),
        ]

        for stage_name, resource, base_duration in stages:
            queue_entry_time = env.now
            queue_len = len(resource.queue)
            self.stage_data[stage_name]["queue_history"].append((env.now, queue_len))

            with resource.request() as req:
                yield req
                wait_duration = env.now - queue_entry_time
                self.stage_data[stage_name]["wait_times"].append(wait_duration)

                # Processing time with slight realistic variability (+- 10%)
                var_factor = random.uniform(0.9, 1.1)
                proc_duration = base_duration * var_factor

                yield env.timeout(proc_duration)
                self.stage_data[stage_name]["busy_time"] += proc_duration
                self.stage_data[stage_name]["processed_count"] += 1

                self.event_log.append(
                    {
                        "Time_Hrs": round(env.now, 2),
                        "Batch_ID": batch_id,
                        "Stage": stage_name,
                        "Wait_Time_Hrs": round(wait_duration, 2),
                        "Proc_Time_Hrs": round(proc_duration, 2),
                    }
                )

        # Quality Inspection Decision
        self.total_produced_yds += self.batch_size_yds
        is_rejected = random.random() < self.defect_prob
        if is_rejected:
            self.batches_rejected += 1
            self.rejected_yds += self.batch_size_yds
        else:
            self.batches_completed += 1
            self.completed_yds += self.batch_size_yds

    def _batch_generator(self, env, resources):
        """Generates continuous raw material batches into the factory."""
        batch_idx = 1
        while True:
            # Exponential inter-arrival time
            inter_arrival = random.expovariate(1.0 / max(0.1, self.arrival_interval_hours))
            yield env.timeout(inter_arrival)
            env.process(self._batch_process(env, f"Batch-{batch_idx:04d}", resources))
            batch_idx += 1

    def run(self):
        """Runs the discrete-event simulation."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        env = simpy.Environment()
        resources = {
            "Warping": simpy.Resource(env, capacity=self.warping_machines),
            "Sizing": simpy.Resource(env, capacity=self.sizing_machines),
            "Weaving": simpy.Resource(env, capacity=self.weaving_machines),
            "Inspection": simpy.Resource(env, capacity=self.inspection_machines),
        }

        # Start batch generator
        env.process(self._batch_generator(env, resources))
        env.run(until=self.sim_duration_hours)

        return self.get_results()

    def get_results(self):
        """Compiles simulation KPIs, utilization rates, and waiting statistics."""
        stage_metrics = {}
        capacities = {
            "Warping": self.warping_machines,
            "Sizing": self.sizing_machines,
            "Weaving": self.weaving_machines,
            "Inspection": self.inspection_machines,
        }

        for stage, data in self.stage_data.items():
            cap = capacities[stage]
            total_available_machine_hours = cap * self.sim_duration_hours
            utilization = (data["busy_time"] / max(0.001, total_available_machine_hours)) * 100.0
            utilization = min(100.0, round(utilization, 2))

            avg_wait = np.mean(data["wait_times"]) if data["wait_times"] else 0.0
            max_wait = np.max(data["wait_times"]) if data["wait_times"] else 0.0

            stage_metrics[stage] = {
                "machines": cap,
                "utilization_pct": utilization,
                "avg_wait_hours": round(float(avg_wait), 2),
                "max_wait_hours": round(float(max_wait), 2),
                "processed_count": data["processed_count"],
            }

        rejection_rate_pct = (
            (self.batches_rejected / max(1, (self.batches_completed + self.batches_rejected))) * 100.0
        )
        throughput_yds_per_hr = self.completed_yds / max(0.1, self.sim_duration_hours)

        return {
            "sim_duration_hours": self.sim_duration_hours,
            "batches_started": self.batches_started,
            "batches_completed": self.batches_completed,
            "batches_rejected": self.batches_rejected,
            "total_produced_yds": round(self.total_produced_yds, 1),
            "completed_yds": round(self.completed_yds, 1),
            "rejected_yds": round(self.rejected_yds, 1),
            "rejection_rate_pct": round(rejection_rate_pct, 2),
            "throughput_yds_per_hr": round(throughput_yds_per_hr, 2),
            "stage_metrics": stage_metrics,
            "event_log_df": pd.DataFrame(self.event_log),
        }
