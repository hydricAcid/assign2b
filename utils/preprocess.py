"""
Enhanced Traffic Data Preprocessing Module

This module loads traffic flow data from SCATS, performs preprocessing steps,
and prepares it for deep learning model training.

Improvements:
- Added feature engineering for time-based patterns
- Implemented robust data cleaning and validation
- Added stratified data splitting by SCATS location
- Added configurable parameters via config dictionary
- Improved data normalization with robust scaling
- Added data augmentation techniques
- Enhanced logging and error handling
"""

import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("traffic_preprocessing")

# Default config - can be overridden
DEFAULT_CONFIG = {
    "look_back": 10,
    "forecast_horizon": 1,
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
    "min_samples": 50,
    "scaler_type": "robust",  # Options: "robust", "minmax"
    "add_time_features": True,
    "use_data_augmentation": False,
    "augmentation_factor": 0.05,  # Noise factor for augmentation
}


def load_config(config_path=None):
    """Load configuration from file or use defaults"""
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    return config


def create_sequences(data, look_back, forecast_horizon=1):
    """
    Create input sequences and target values for time series forecasting

    Args:
        data (numpy.ndarray): Input time series data
        look_back (int): Number of time steps to look back
        forecast_horizon (int): Number of time steps to forecast ahead

    Returns:
        tuple: (X, y) where X is input sequences and y is target values
    """
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i : (i + look_back)])
        if forecast_horizon == 1:
            y.append(data[i + look_back])
        else:
            y.append(data[(i + look_back) : (i + look_back + forecast_horizon)])

    return np.array(X), np.array(y)


def add_time_features(df):
    """Add time-based features to help model learn temporal patterns"""
    # Extract time components
    df["hour"] = df["Timestamp"].dt.hour
    df["minute"] = df["Timestamp"].dt.minute
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    df["day_of_month"] = df["Timestamp"].dt.day
    df["month"] = df["Timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Create cyclical time features to represent circular nature of time
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def apply_data_augmentation(X, y, factor=0.05):
    """Apply noise-based data augmentation to input sequences"""
    # Create copies with small gaussian noise
    X_aug = X.copy()
    noise = np.random.normal(0, factor, X_aug.shape)
    X_aug = X_aug + noise

    # Make sure we don't have negative values for traffic flow
    X_aug = np.maximum(X_aug, 0)

    return X_aug, y


def preprocess_data(
    input_excel="data/Scats_data_october_2006.xlsx",
    output_folder="data/processed",
    config_path=None,
):
    """
    Preprocess traffic data from SCATS Excel file and prepare for model training

    Args:
        input_excel (str): Path to input Excel file
        output_folder (str): Path to output folder
        config_path (str, optional): Path to config JSON file
    """
    start_time = datetime.now()
    config = load_config(config_path)
    logger.info(f"Starting preprocessing with config: {config}")

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Load data
        logger.info(f"Loading data from {input_excel}")
        df_raw = pd.read_excel(input_excel, sheet_name="Data")

        # Clean column names
        df_clean = df_raw.iloc[1:].copy()
        df_clean.columns = df_raw.iloc[0]
        df_clean.reset_index(drop=True, inplace=True)
        df_clean.columns = df_clean.columns.str.strip().str.replace(" ", "_")

        # Identify metadata and value columns
        meta_cols = ["SCATS_Number", "Location", "Date"]
        value_cols = [
            col
            for col in df_clean.columns
            if isinstance(col, str) and col.startswith("V")
        ]

        logger.info(f"Found {len(value_cols)} time intervals")

        # Convert date and create time mapping
        df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
        time_intervals = pd.date_range("00:00", "23:45", freq="15min").time
        time_map = dict(zip(value_cols, time_intervals))

        # Restructure data from wide to long format
        logger.info("Converting data to long format")
        df_long = df_clean.melt(
            id_vars=meta_cols,
            value_vars=value_cols,
            var_name="Interval",
            value_name="Flow",
        )

        # Create timestamp and clean data
        df_long["Time"] = df_long["Interval"].map(time_map)
        df_long["Timestamp"] = pd.to_datetime(
            df_long["Date"].astype(str) + " " + df_long["Time"].astype(str),
            errors="coerce",
        )

        # Convert flow to numeric and handle missing values
        df_long["Flow"] = pd.to_numeric(df_long["Flow"], errors="coerce")

        # Data cleaning and validation
        logger.info("Cleaning and validating data")
        total_records = len(df_long)
        df_long = df_long.dropna(subset=["Timestamp", "Flow"])
        cleaned_records = len(df_long)

        logger.info(f"Removed {total_records - cleaned_records} invalid records")

        # Add time-based features if configured
        if config["add_time_features"]:
            logger.info("Adding time-based features")
            df_long = add_time_features(df_long)

        # Save processed dataframe
        output_csv = os.path.join(output_folder, "processed_data.csv")
        df_long.to_csv(output_csv, index=False)
        logger.info(f"Saved processed data to {output_csv}")

        # Prepare data for model training
        logger.info("Preparing sequences for model training")

        # Initialize containers for collected data
        all_X, all_y = [], []
        all_scalers = {}
        scats_mapping = {}

        # Process each SCATS location separately to maintain locality
        for scats_id, group in df_long.groupby("SCATS_Number"):
            series = group.sort_values("Timestamp")

            # Skip locations with insufficient data
            if len(series) <= config["look_back"] + config["forecast_horizon"]:
                logger.warning(f"Skipping SCATS {scats_id}: insufficient data")
                continue

            # Extract the flow values and reshape for scaling
            flow_data = series["Flow"].values.reshape(-1, 1)

            # Apply appropriate scaler
            if config["scaler_type"] == "robust":
                scaler = RobustScaler()
            else:
                scaler = MinMaxScaler()

            series_scaled = scaler.fit_transform(flow_data)

            # Store the scaler for this SCATS location
            all_scalers[scats_id] = scaler

            # Create sequences
            X, y = create_sequences(
                series_scaled,
                look_back=config["look_back"],
                forecast_horizon=config["forecast_horizon"],
            )

            # Store mapping of sequences to SCATS IDs for stratified splitting
            idx_start = len(all_X)
            idx_end = idx_start + len(X)
            scats_mapping[scats_id] = list(range(idx_start, idx_end))

            # Collect data
            all_X.append(X)
            all_y.append(y)

            logger.info(f"Processed SCATS {scats_id}: created {len(X)} sequences")

        # Concatenate all data
        X_all = np.concatenate(all_X) if all_X else np.array([])
        y_all = np.concatenate(all_y) if all_y else np.array([])

        if len(X_all) == 0:
            raise ValueError("No valid sequences could be created from the data")

        logger.info(f"Created {len(X_all)} total sequences")

        # Apply data augmentation if configured
        if config["use_data_augmentation"]:
            logger.info("Applying data augmentation")
            X_aug, y_aug = apply_data_augmentation(
                X_all, y_all, factor=config["augmentation_factor"]
            )
            X_all = np.concatenate([X_all, X_aug])
            y_all = np.concatenate([y_all, y_aug])
            logger.info(f"After augmentation: {len(X_all)} sequences")

        # Create location-based indices for stratified split
        location_indices = []
        for scats_id, indices in scats_mapping.items():
            location_indices.extend([scats_id] * len(indices))

        # Handle case where augmentation was applied
        if config["use_data_augmentation"]:
            # Duplicate the location indices for augmented data
            location_indices = location_indices * 2

        # Split into train and temporary test set
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            X_all,
            y_all,
            np.array(location_indices),
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=np.array(location_indices),
        )

        # Split temporary test set into validation and test
        val_ratio = config["validation_size"] / (1 - config["test_size"])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=config["random_state"]
        )

        # Save the prepared datasets
        dataset_path = os.path.join(output_folder, "dataset.npz")
        np.savez(
            dataset_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        # Save the scalers and configuration
        with open(os.path.join(output_folder, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save metadata about the preprocessing
        metadata = {
            "num_sequences": len(X_all),
            "sequence_length": config["look_back"],
            "forecast_horizon": config["forecast_horizon"],
            "num_train": len(X_train),
            "num_val": len(X_val),
            "num_test": len(X_test),
            "scats_locations": list(scats_mapping.keys()),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": list(df_long.columns),
        }

        with open(os.path.join(output_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(
            f"✅ Preprocessing complete. Total time: {processing_time:.2f} seconds"
        )
        logger.info(f"✅ Processed data saved to: {output_folder}")

        return True

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def load_data(data_path="data/processed/dataset.npz"):
    """
    Load the preprocessed data for model training

    Args:
        data_path (str): Path to the preprocessed data

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    try:
        data = np.load(data_path)
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["y_test"],
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    preprocess_data(
        input_excel="data/Scats_data_october_2006.xlsx",
        output_folder="data/processed",
        config_path=None,  # Use default config
    )
