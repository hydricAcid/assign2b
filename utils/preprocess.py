import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime
import logging
from joblib import dump

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - traffic_preprocessing - %(levelname)s - %(message)s",
)
logger = logging.getLogger("traffic_preprocessing")


def create_sequences(series, look_back=10, forecast_horizon=1):
    X, y = [], []
    for i in range(len(series) - look_back - forecast_horizon + 1):
        X.append(series[i : i + look_back])
        y.append(series[i + look_back + forecast_horizon - 1])
    return np.array(X), np.array(y)


def preprocess_data():
    config = {
        "look_back": 10,
        "forecast_horizon": 1,
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42,
        "min_samples": 50,
        "scaler_type": "robust",
        "add_time_features": True,
        "use_data_augmentation": False,
        "augmentation_factor": 0.05,
    }

    logger.info(f"Starting preprocessing with config: {config}")

    file_path = "data/raw/Scats_data_october_2006.xlsx"
    df = pd.read_excel(file_path, sheet_name="Data", header=1)
    df.columns = df.columns.astype(str).str.strip()

    print("✅ Standard column after header reset:")
    print(df.columns.tolist())

    df = df.rename(columns={"Date": "Timestamp"})

    df_long = df.melt(
        id_vars=["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE", "Timestamp"],
        var_name="Interval",
        value_name="Flow",
    )

    df_long["Timestamp"] = pd.to_datetime(df_long["Timestamp"], errors="coerce")
    df_long = df_long.dropna(subset=["Timestamp", "Flow"])
    logger.info(f"Removed {df.shape[0] - df_long.shape[0]} invalid records")

    if config["add_time_features"]:
        logger.info("Adding time-based features")
        df_long["hour"] = df_long["Timestamp"].dt.hour
        df_long["dayofweek"] = df_long["Timestamp"].dt.dayofweek

    df_long["Flow"] = pd.to_numeric(df_long["Flow"], errors="coerce")
    df_long = df_long.dropna(subset=["Flow"])
    df_long = df_long[df_long["Flow"] >= 0]

    os.makedirs("data/processed", exist_ok=True)
    df_long.to_csv("data/processed/processed_data.csv", index=False)
    logger.info("Saved processed data to data/processed/processed_data.csv")

    logger.info("Preparing sequences for model training...")

    X_all, y_all, site_ids_all = [], [], []

    for scats_id, df_group in df_long.groupby("SCATS Number"):
        df_group = df_group.sort_values("Timestamp")
        series = df_group["Flow"].values

        if len(series) < config["look_back"] + config["forecast_horizon"]:
            continue

        X, y = create_sequences(series, config["look_back"], config["forecast_horizon"])

        X_all.append(X)
        y_all.append(y)
        site_ids_all += [str(scats_id)] * len(y)

        logger.info(f"Processed SCATS {scats_id}: created {len(y)} sequences")

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    site_ids_all = np.array(site_ids_all)

    scaler_X = RobustScaler() if config["scaler_type"] == "robust" else MinMaxScaler()
    scaler_y = RobustScaler() if config["scaler_type"] == "robust" else MinMaxScaler()

    X_shape = X_all.shape
    X_all_scaled = scaler_X.fit_transform(X_all.reshape(-1, X_shape[-1])).reshape(
        X_shape
    )
    y_all_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

    dump(scaler_y, "data/processed/output_scaler.pkl")
    logger.info("✅ Output scaler saved to data/processed/output_scaler.pkl")

    rng = np.random.RandomState(config["random_state"])
    indices = rng.permutation(len(X_all_scaled))

    X_all_scaled = X_all_scaled[indices]
    y_all_scaled = y_all_scaled[indices]
    site_ids_all = site_ids_all[indices]

    N = len(X_all_scaled)
    test_size = int(config["test_size"] * N)
    val_size = int(config["validation_size"] * N)

    X_test, y_test, site_ids_test = (
        X_all_scaled[:test_size],
        y_all_scaled[:test_size],
        site_ids_all[:test_size],
    )
    X_val, y_val, site_ids_val = (
        X_all_scaled[test_size : test_size + val_size],
        y_all_scaled[test_size : test_size + val_size],
        site_ids_all[test_size : test_size + val_size],
    )
    X_train, y_train, site_ids_train = (
        X_all_scaled[test_size + val_size :],
        y_all_scaled[test_size + val_size :],
        site_ids_all[test_size + val_size :],
    )

    np.savez(
        "data/processed/dataset.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        site_ids_train=site_ids_train,
        site_ids_val=site_ids_val,
        site_ids_test=site_ids_test,
    )

    logger.info("✅ Preprocessing complete. Total sequences: %d", len(X_all_scaled))


if __name__ == "__main__":
    preprocess_data()
