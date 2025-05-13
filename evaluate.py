import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from scipy import stats
import tensorflow as tf
from keras.api.models import load_model
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


class ModelEvaluator:
    def __init__(
        self, data_path="data/processed/dataset.npz", output_dir="evaluation_results"
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.ensemble_predictions = None

        os.makedirs(output_dir, exist_ok=True)

        try:
            self.load_data()
            print(f"✅ Test data loaded successfully: {self.X_test.shape[0]} samples")
        except Exception as e:
            print(f"❌ Error loading test data: {e}")

    def load_data(self):
        data = np.load(self.data_path)
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

        try:
            metadata_path = os.path.join(
                os.path.dirname(self.data_path), "metadata.json"
            )
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.has_time_data = True
        except:
            self.metadata = None
            self.has_time_data = False

    def load_model(self, model_path, model_name=None):
        try:
            if model_name is None:
                model_name = os.path.basename(model_path).split(".")[0]

            self.models[model_name] = load_model(
                model_path,
            )
            print(f"✅ Model '{model_name}' loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model '{model_name}' from {model_path}: {e}")
            return False

    def generate_predictions(self, model_name=None):
        models_to_predict = [model_name] if model_name else self.models.keys()

        for name in models_to_predict:
            if name in self.models:
                start_time = time.time()
                self.predictions[name] = self.models[name].predict(
                    self.X_test, batch_size=128
                )
                pred_time = time.time() - start_time
                print(
                    f"✅ Generated predictions for model '{name}' in {pred_time:.2f} seconds"
                )
            else:
                print(f"❌ Model '{name}' not found")

    def calculate_metrics(self, model_name=None):
        models_to_evaluate = [model_name] if model_name else self.predictions.keys()

        for name in models_to_evaluate:
            if name in self.predictions:
                y_pred = self.predictions[name]

                metrics = {
                    "mse": mean_squared_error(self.y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    "mae": mean_absolute_error(self.y_test, y_pred),
                    "mape": np.mean(
                        np.abs(
                            (self.y_test - y_pred)
                            / np.clip(np.abs(self.y_test), 1e-5, None)
                        )
                    ),
                    "r2": r2_score(self.y_test, y_pred),
                }

                self.metrics[name] = metrics
                print(f"📊 Metrics for model '{name}':")
                for k, v in metrics.items():
                    print(f"  {k.upper()}: {v:.4f}")
            else:
                print(f"❌ Predictions not found for model '{name}'")

    def create_ensemble(self, method="average"):
        if not self.predictions:
            print("❌ No predictions available for ensemble")
            return

        all_preds = np.array([pred for pred in self.predictions.values()])

        if method == "average":
            self.ensemble_predictions = np.mean(all_preds, axis=0)
        elif method == "median":
            self.ensemble_predictions = np.median(all_preds, axis=0)
        elif method == "min":
            self.ensemble_predictions = np.min(all_preds, axis=0)
        elif method == "max":
            self.ensemble_predictions = np.max(all_preds, axis=0)

        self.predictions["ensemble"] = self.ensemble_predictions
        print(f"✅ Created ensemble predictions using {method} method")

    def compare_models(self):
        if not self.metrics:
            print("❌ No metrics available for comparison")
            return

        comparison_df = pd.DataFrame.from_dict(self.metrics, orient="index")

        output_path = os.path.join(self.output_dir, "model_comparison.csv")
        comparison_df.to_csv(output_path)
        print(f"✅ Saved model comparison to {output_path}")

        return comparison_df

    def plot_predictions(self, model_name=None, samples=20):
        if model_name is None and self.ensemble_predictions is not None:
            y_pred = self.ensemble_predictions
            title = "Ensemble Predictions"
        elif model_name in self.predictions:
            y_pred = self.predictions[model_name]
            title = f"{model_name} Predictions"
        else:
            print("❌ No valid predictions to plot")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.y_test[:samples], label="Actual", marker="o")
        plt.plot(y_pred[:samples], label="Predicted", marker="x")
        plt.title(f"Actual vs Predicted Values\n{title}")
        plt.xlabel("Sample Index")
        plt.ylabel("Traffic Flow")
        plt.legend()

        output_path = os.path.join(
            self.output_dir, f"{title.lower().replace(' ', '_')}.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved prediction plot to {output_path}")

    def plot_error_distribution(self, model_name):
        if model_name not in self.predictions:
            print(f"❌ Model '{model_name}' not found")
            return

        errors = self.y_test - self.predictions[model_name].flatten()

        plt.figure(figsize=(12, 6))
        sns.histplot(errors, kde=True)
        plt.title(f"Error Distribution for {model_name}")
        plt.xlabel("Prediction Error")

        output_path = os.path.join(self.output_dir, f"{model_name}_error_dist.png")
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved error distribution plot to {output_path}")

    def time_based_evaluation(self, model_name):
        if not self.has_time_data:
            print("❌ No time metadata available for time-based evaluation")
            return

        if model_name not in self.predictions:
            print(f"❌ Model '{model_name}' not found")
            return

        results_df = pd.DataFrame(
            {
                "actual": self.y_test.flatten(),
                "predicted": self.predictions[model_name].flatten(),
            }
        )

        time_features = ["hour", "day_of_week", "is_weekend"]
        for feat in time_features:
            if feat in self.metadata["features"]:
                results_df[feat] = self.X_test[
                    :, -1, self.metadata["features"].index(feat)
                ]

        results_df["error"] = results_df["actual"] - results_df["predicted"]
        results_df["abs_error"] = np.abs(results_df["error"])

        time_metrics = {}
        for feat in time_features:
            if feat in results_df:
                grouped = results_df.groupby(feat)["abs_error"].mean()
                time_metrics[feat] = grouped

                plt.figure()
                grouped.plot(kind="bar")
                plt.title(f"MAE by {feat.replace('_', ' ').title()}")
                plt.ylabel("Mean Absolute Error")

                output_path = os.path.join(
                    self.output_dir, f"{model_name}_performance_by_{feat}.png"
                )
                plt.savefig(output_path)
                plt.close()
                print(f"✅ Saved {feat} performance plot to {output_path}")

        return time_metrics

    def save_results(self):
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)

        if self.ensemble_predictions is not None:
            np.save(
                os.path.join(self.output_dir, "ensemble_predictions.npy"),
                self.ensemble_predictions,
            )

        print(f"✅ All results saved to {self.output_dir}")


if __name__ == "__main__":
    evaluator = ModelEvaluator()

    model_paths = {
        "LSTM": "models/lstm/best_model.keras",
        "GRU": "models/gru/best_model.keras",
        "CNN": "models/cnn/best_model.keras",
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            evaluator.load_model(path, name)
        else:
            print(f"⚠️ Skipping {name} – model file not found at {path}")

    evaluator.generate_predictions()
    evaluator.calculate_metrics()

    evaluator.create_ensemble()
    evaluator.calculate_metrics("ensemble")

    evaluator.plot_predictions("LSTM")
    evaluator.plot_error_distribution("LSTM")
    evaluator.time_based_evaluation("LSTM")

    evaluator.compare_models()
    evaluator.save_results()
