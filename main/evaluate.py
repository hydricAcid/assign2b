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
from utils.convert_to_travel_time import flow_to_travel_time


def evaluate_model(
    model_path,
    dataset_path,
    scaler_path,
    output_folder="evaluation_results",
    distance_km=1.0,
):
    os.makedirs(output_folder, exist_ok=True)

    data = np.load(dataset_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    site_ids = data["site_ids_test"]
    timestamps = data["timestamps_test"]

    model = load_model(model_path)

    y_pred = model.predict(X_test).flatten()

    scaler = load(scaler_path)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    travel_time_pred = flow_to_travel_time(y_pred, timestamps, distance_km=distance_km)
    travel_time_true = flow_to_travel_time(y_true, timestamps, distance_km=distance_km)

    mae = mean_absolute_error(travel_time_true, travel_time_pred)
    mse = mean_squared_error(travel_time_true, travel_time_pred)
    r2 = r2_score(travel_time_true, travel_time_pred)

    metrics = {"test_mae": mae, "test_loss": mse, "test_r2": r2}
    with open(os.path.join(output_folder, "evaluation.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    df = pd.DataFrame(
        {
            "SCATS_ID": site_ids,
            "Timestamp": timestamps,
            "Actual": y_true,
            "Predicted": y_pred,
            "TravelTime": travel_time_pred,
        }
    )
    df.to_csv(os.path.join(output_folder, "CNN_predictions.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:200], label="True")
    plt.plot(y_pred[:200], label="Predicted")
    plt.legend()
    plt.title("Prediction vs True Flow")
    plt.savefig(os.path.join(output_folder, "CNN_plot.png"))
    plt.close()

    print("✅ Evaluation complete. MAE:", mae)


if __name__ == "__main__":
    evaluate_model(
        model_path="models/cnn_20250524_004759/best_model.keras",
        dataset_path="data/processed/dataset.npz",
        scaler_path="data/processed/output_scaler.pkl",
        distance_km=1.0,
    )
