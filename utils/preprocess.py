import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
import joblib


config = {
    "look_back": 10,
    "forecast_horizon": 1,
    "output_dir": "data/processed"
}


df = pd.read_excel("data/raw/Scats_data_october_2006.xlsx", sheet_name="Data", header=1)
df.columns = df.columns.astype(str).str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


interval_columns = [f"V{str(i).zfill(2)}" for i in range(1, 97)]
interval_columns = [col for col in interval_columns if col in df.columns]

df_long = df.melt(
    id_vars=["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE", "Date"],
    value_vars=interval_columns,
    var_name="Interval",
    value_name="Flow"
)

df_long = df_long[~((df_long["NB_LATITUDE"] == 0) & (df_long["NB_LONGITUDE"] == 0))]

def interval_to_minutes(interval): return int(interval[1:]) * 15
df_long["TimeOffset"] = df_long["Interval"].apply(interval_to_minutes)
df_long["Timestamp"] = df_long["Date"] + pd.to_timedelta(df_long["TimeOffset"], unit="m")
df_long = df_long.dropna(subset=["Timestamp"])

df_long["hour"] = df_long["Timestamp"].dt.hour
df_long["dayofweek"] = df_long["Timestamp"].dt.dayofweek
df_long["month"] = df_long["Timestamp"].dt.month
df_long["day"] = df_long["Timestamp"].dt.day

df_long.rename(columns={"SCATS Number": "SCATS_ID"}, inplace=True)
df_long = df_long[["SCATS_ID", "Timestamp", "Flow", "hour", "dayofweek", "month", "day"]]

X, y, site_ids, timestamps = [], [], [], []

for site_id, group in df_long.groupby("SCATS_ID"):
    group = group.sort_values("Timestamp").reset_index(drop=True)
    features = group[["Flow", "hour", "dayofweek", "month", "day"]].values.astype(np.float32)

    for i in range(len(features) - config["look_back"] - config["forecast_horizon"] + 1):
        X.append(features[i:i+config["look_back"]])
        y.append(features[i+config["look_back"]][0])
        site_ids.append(site_id)
        timestamps.append(group.loc[i+config["look_back"], "Timestamp"])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
site_ids = np.array(site_ids)
timestamps = np.array(timestamps)

flow_scaler = RobustScaler()
y_scaled = flow_scaler.fit_transform(y.reshape(-1, 1)).flatten()

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
site_ids_test = site_ids[split_index:]
timestamps_test = timestamps[split_index:]

os.makedirs(config["output_dir"], exist_ok=True)
np.savez_compressed(
    os.path.join(config["output_dir"], "dataset.npz"),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    site_ids_test=site_ids_test,
    timestamps_test=timestamps_test
)
joblib.dump(flow_scaler, os.path.join(config["output_dir"], "output_scaler.pkl"))

print("✅ Data preprocessing complete. Saved to:", config["output_dir"])
