import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from datetime import timedelta
import logging
from joblib import dump

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - traffic_preprocessing - %(levelname)s - %(message)s",
)
logger = logging.getLogger("traffic_preprocessing")


def preprocess_data():
    config = {
        "look_back": 9,
        "forecast_horizon": 1,
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42,
    }

    logger.info(f"Starting preprocessing with config: {config}")

    file_path = "data/raw/Scats_data_october_2006.xlsx"
    df = pd.read_excel(file_path, sheet_name="Data", header=1)
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns={"Date": "Timestamp"})

    df_long = df.melt(
        id_vars=["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE", "Timestamp"],
        var_name="Interval",
        value_name="Flow",
    )

    df_long = df_long[(df_long["NB_LATITUDE"] != 0) & (df_long["NB_LONGITUDE"] != 0)]
    df_long = df_long[df_long["Interval"].str.match(r"^V\d{2}$")]

    def interval_to_time(iv):
        idx = int(iv[1:])
        return timedelta(minutes=15 * idx)

    df_long["Time"] = df_long["Interval"].apply(interval_to_time)
    df_long["Timestamp"] = (
        pd.to_datetime(df_long["Timestamp"], errors="coerce") + df_long["Time"]
    )

    df_long = df_long.dropna(subset=["Timestamp", "Flow"])
    df_long["Flow"] = pd.to_numeric(df_long["Flow"], errors="coerce")
    df_long = df_long.dropna(subset=["Flow"])
    df_long = df_long[df_long["Flow"] >= 0]

    df_long["hour"] = df_long["Timestamp"].dt.hour

    avg_coords = (
        df_long.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]]
        .mean()
        .reset_index()
        .rename(columns={"NB_LATITUDE": "Avg_LAT", "NB_LONGITUDE": "Avg_LON"})
    )
    avg_coords.to_csv("data/doc/nodes_averaged.txt", index=False)
    df_long = df_long.merge(avg_coords, on="SCATS Number", how="left")

    os.makedirs("data/processed", exist_ok=True)
    df_long.to_csv("data/processed/processed_data.csv", index=False)

    logger.info("Preparing sequences for model training...")

    X_all, y_all, site_ids_all = [], [], []

    for scats_id, df_group in df_long.groupby("SCATS Number"):
        df_group = df_group.sort_values("Timestamp")
        df_group = df_group[["Flow", "hour"]].copy()
        data = df_group.values.astype(np.float32)

        if len(data) < config["look_back"] + config["forecast_horizon"]:
            continue

        X, y = [], []
        for i in range(
            len(data) - config["look_back"] - config["forecast_horizon"] + 1
        ):
            window = data[i : i + config["look_back"]]
            target = data[i + config["look_back"] + config["forecast_horizon"] - 1][0]
            X.append(window)
            y.append(target)

        X_all.append(np.array(X, dtype=np.float32))
        y_all.append(np.array(y, dtype=np.float32))
        site_ids_all += [int(scats_id)] * len(y)

        logger.info(f"Processed SCATS {scats_id}: {len(y)} sequences")

    X_all = np.concatenate(X_all).astype(np.float32)
    y_all = np.concatenate(y_all).astype(np.float32)
    site_ids_all = np.array(site_ids_all, dtype=np.int32)

    # Scale flow only (feature 0)
    scaler_X = RobustScaler()
    X_all[:, :, 0] = scaler_X.fit_transform(X_all[:, :, 0])

    scaler_y = RobustScaler()
    y_all_scaled = (
        scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten().astype(np.float32)
    )
    dump(scaler_y, "data/processed/output_scaler.pkl")

    rng = np.random.RandomState(config["random_state"])
    indices = rng.permutation(len(X_all))

    X_all = X_all[indices]
    y_all_scaled = y_all_scaled[indices]
    site_ids_all = site_ids_all[indices]

    N = len(X_all)
    test_size = int(config["test_size"] * N)
    val_size = int(config["validation_size"] * N)

    X_test = X_all[:test_size]
    y_test = y_all_scaled[:test_size]
    site_ids_test = site_ids_all[:test_size]

    X_val = X_all[test_size : test_size + val_size]
    y_val = y_all_scaled[test_size : test_size + val_size]
    site_ids_val = site_ids_all[test_size : test_size + val_size]

    X_train = X_all[test_size + val_size :]
    y_train = y_all_scaled[test_size + val_size :]
    site_ids_train = site_ids_all[test_size + val_size :]

    np.savez_compressed(
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

    logger.info("✅ Preprocessing complete. Total sequences: %d", len(X_all))


def load_data(npz_path):
    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    preprocess_data()
