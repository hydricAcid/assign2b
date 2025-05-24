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
    distance_km=5.0,
):
    os.makedirs(output_folder, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    site_ids = data["site_ids"]
    timestamps = data["timestamps"]

    model = load_model(model_path)

    y_pred = model.predict(X).flatten()

    scaler = load(scaler_path)
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    travel_time_pred = flow_to_travel_time(y_pred, timestamps, distance_km=distance_km)
    travel_time_true = flow_to_travel_time(y_true, timestamps, distance_km=distance_km)

    mae = mean_absolute_error(travel_time_true, travel_time_pred)
    mse = mean_squared_error(travel_time_true, travel_time_pred)
    r2 = r2_score(travel_time_true, travel_time_pred)

    metrics = {"mae": mae, "mse": mse, "r2": r2}
    with open(os.path.join(output_folder, "evaluation_full.json"), "w") as f:
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

    nodes_df = pd.read_csv("data/doc/nodes_averaged.txt")
    df["SCATS_ID"] = df["SCATS_ID"].astype(int)
    nodes_df["SCATS_ID"] = nodes_df["SCATS_ID"].astype(int)
    df = df.merge(nodes_df, on="SCATS_ID", how="left")

    output_file = os.path.join(output_folder, "CNN_predictions_full.csv")
    df.to_csv(output_file, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:200], label="True")
    plt.plot(y_pred[:200], label="Predicted")
    plt.legend()
    plt.title("Prediction vs True Flow (Full Data)")
    plt.savefig(os.path.join(output_folder, "CNN_plot_full.png"))
    plt.close()

    print("✅ Full Evaluation complete. MAE:", mae)


if __name__ == "__main__":
    evaluate_model(
        model_path="models/cnn_20250524_215156/best_model.keras",
        dataset_path="data/processed/dataset.npz",
        scaler_path="data/processed/output_scaler.pkl",
        distance_km=5.0,
    )
