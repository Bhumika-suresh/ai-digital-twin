"""
Machine Learning Model module for Textile Rejection & Production Prediction.
Uses Scikit-learn RandomForest with real training, metrics, and feature importance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class TextileMLSystem:
    def __init__(self):
        self.clf_model = None
        self.reg_model = None
        self.scaler = None
        self.feature_names = []
        self.clf_metrics = {}
        self.reg_metrics = {}
        self.feature_importances_clf = {}
        self.feature_importances_reg = {}
        self.is_trained = False

    def select_features_and_targets(self, df):
        """
        Dynamically extracts features, classification target, and regression target from the dataframe.
        """
        # Exclude targets and identifier columns
        exclude_cols = [
            "Rejection",
            "Rejection_Qty",
            "Has_Rejection",
            "Rej_and_cut_Piece",
            "Total_Pdn(yds)",
            "Total_pdn_per_order",
            "Total_pdn_m/c",
            "ID",
            "Month",
            "Construction",
            "Previous_pdn",
            "warp_count",
            "act_crimp%",
        ]

        # Candidate numeric features
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
        
        # Select best available numeric columns
        if not num_cols:
            raise ValueError("No numeric features found for ML training.")

        X = df[num_cols].copy().fillna(0)

        # Targets
        y_clf = None
        if "Has_Rejection" in df.columns:
            y_clf = df["Has_Rejection"].values
        elif "Rejection" in df.columns and pd.api.types.is_numeric_dtype(df["Rejection"]):
            y_clf = (df["Rejection"] > 0).astype(int).values

        y_reg = None
        if "Total_Pdn(yds)" in df.columns:
            y_reg = df["Total_Pdn(yds)"].values
        elif "Total_pdn_per_order" in df.columns:
            y_reg = df["Total_pdn_per_order"].values
        elif "Rejection_Qty" in df.columns:
            y_reg = df["Rejection_Qty"].values

        return X, y_clf, y_reg, num_cols

    def train_models(self, df, test_size=0.2, random_state=42):
        """
        Trains RandomForestClassifier and RandomForestRegressor on the provided dataset.
        Calculates and stores real performance metrics and feature importances.
        """
        X, y_clf, y_reg, feature_names = self.select_features_and_targets(df)
        self.feature_names = feature_names

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 1. Train Classification Model (Rejection Risk)
        if y_clf is not None and len(np.unique(y_clf)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
            )
            self.clf_model = RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1
            )
            self.clf_model.fit(X_train, y_train)

            y_pred = self.clf_model.predict(X_test)
            y_prob = self.clf_model.predict_proba(X_test)[:, 1] if hasattr(self.clf_model, "predict_proba") else y_pred

            cm = confusion_matrix(y_test, y_pred)
            self.clf_metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "confusion_matrix": cm.tolist(),
                "test_samples": len(y_test),
                "train_samples": len(y_train),
            }

            # Feature importances
            importances = self.clf_model.feature_importances_
            self.feature_importances_clf = dict(
                sorted(zip(self.feature_names, [float(i) for i in importances]), key=lambda x: x[1], reverse=True)
            )

        # 2. Train Regression Model (Production Volume / Yardage)
        if y_reg is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_reg, test_size=test_size, random_state=random_state
            )
            self.reg_model = RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1
            )
            self.reg_model.fit(X_train, y_train)

            y_pred = self.reg_model.predict(X_test)
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            self.reg_metrics = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "test_samples": len(y_test),
                "train_samples": len(y_train),
            }

            importances = self.reg_model.feature_importances_
            self.feature_importances_reg = dict(
                sorted(zip(self.feature_names, [float(i) for i in importances]), key=lambda x: x[1], reverse=True)
            )

        self.is_trained = True
        self.save_models()
        return self.clf_metrics, self.reg_metrics

    def predict_single(self, input_dict):
        """
        Makes a live prediction on user-supplied textile manufacturing parameters.
        Returns:
            dict: predicted rejection class, rejection probability %, risk tier, expected production yardage.
        """
        if not self.is_trained:
            self.load_models()

        if not self.is_trained or not self.feature_names:
            return {"error": "Model not trained yet."}

        # Build feature vector matching feature_names
        row_dict = {}
        for feat in self.feature_names:
            val = input_dict.get(feat, 0.0)
            row_dict[feat] = float(val)

        X_input_df = pd.DataFrame([row_dict], columns=self.feature_names)
        X_scaled = self.scaler.transform(X_input_df)

        res = {}
        if self.clf_model is not None:
            pred_class = int(self.clf_model.predict(X_scaled)[0])
            pred_prob = float(self.clf_model.predict_proba(X_scaled)[0][1]) * 100
            
            if pred_prob < 30:
                risk_tier = "🟢 Low Risk (Optimal)"
            elif pred_prob < 65:
                risk_tier = "🟡 Moderate Risk (Monitor Weaving Quality)"
            else:
                risk_tier = "🔴 High Risk (Likely Rejection / Flaw Expected)"

            res["rejection_class"] = pred_class
            res["rejection_probability"] = round(pred_prob, 2)
            res["risk_tier"] = risk_tier

        if self.reg_model is not None:
            pred_prod = float(self.reg_model.predict(X_scaled)[0])
            res["predicted_production_yds"] = round(max(0.0, pred_prod), 2)

        return res

    def save_models(self):
        """Save model artifacts to models/ directory."""
        try:
            joblib.dump(
                {
                    "clf_model": self.clf_model,
                    "reg_model": self.reg_model,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "clf_metrics": self.clf_metrics,
                    "reg_metrics": self.reg_metrics,
                    "feature_importances_clf": self.feature_importances_clf,
                    "feature_importances_reg": self.feature_importances_reg,
                },
                os.path.join(MODELS_DIR, "textile_models.joblib"),
            )
        except Exception:
            pass

    def load_models(self):
        """Load trained model artifacts if present."""
        path = os.path.join(MODELS_DIR, "textile_models.joblib")
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                self.clf_model = data.get("clf_model")
                self.reg_model = data.get("reg_model")
                self.scaler = data.get("scaler")
                self.feature_names = data.get("feature_names", [])
                self.clf_metrics = data.get("clf_metrics", {})
                self.reg_metrics = data.get("reg_metrics", {})
                self.feature_importances_clf = data.get("feature_importances_clf", {})
                self.feature_importances_reg = data.get("feature_importances_reg", {})
                self.is_trained = True
            except Exception:
                pass
