import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load
from keras.api.models import load_model
from utils.convert_to_travel_time import convert_flow_to_travel_time


def evaluate_model(model_path, name="Model"):
    print(f"\n== Evaluation for {name} ==")

    # Load dataset
    data = np.load("data/processed/dataset.npz", allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    site_ids = data["site_ids_test"]

    # Load output scaler
    scaler_y = load("data/processed/output_scaler.pkl")
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Load model
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred).flatten()

    # Metrics
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = mean_squared_error(y_test_orig, y_pred_orig) ** 0.5
    r2 = r2_score(y_test_orig, y_pred_orig)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Save metrics
    os.makedirs("evaluation_results", exist_ok=True)
    metrics_path = os.path.join("evaluation_results", f"{name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2}, f, indent=2)

    # Save predictions to CSV
    df = pd.DataFrame(
        {
            "SCATS_ID": site_ids,
            "Actual": y_test_orig,
            "Predicted": y_pred_orig,
        }
    )
    df["TravelTime"] = df["Predicted"].apply(convert_flow_to_travel_time)
    csv_path = os.path.join("evaluation_results", f"{name}_predictions.csv")
    df.to_csv(csv_path, index=False)

    # Save plot
    plt.figure(figsize=(12, 4))
    plt.plot(y_test_orig[:100], label="Actual")
    plt.plot(y_pred_orig[:100], label="Predicted")
    plt.title(f"{name} - Actual vs Predicted Flow (veh/hr)")
    plt.xlabel("Sample Index")
    plt.ylabel("Flow")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join("evaluation_results", f"{name}_plot.png")
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    evaluate_model(model_path="models/cnn_20250514_152940/best_model.keras", name="CNN")
