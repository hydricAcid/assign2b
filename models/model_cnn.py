import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Input,
    Add,
    Concatenate,
    Flatten,
    SpatialDropout1D,
    Activation,
)
from keras.api.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.api.optimizers import Adam
from keras.api.regularizers import l1_l2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_data

# Configuration settings
CONFIG = {
    "model_name": "enhanced_cnn",
    "filters": [64, 128, 256],  # Filters in each conv layer
    "kernel_sizes": [3, 5, 7],  # Different kernel sizes for multi-scale
    "dilation_rates": [1, 2, 4],  # Dilation rates for capturing long-range patterns
    "dense_units": [128, 64],  # Units in dense layers
    "dropout_rate": 0.3,
    "spatial_dropout": 0.2,  # Dropout entire feature maps
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "patience": 10,
    "use_residual": True,
    "use_inception": True,
    "use_batch_norm": True,
    "use_layer_norm": False,
    "l1_reg": 0.0001,
    "l2_reg": 0.0001,
}


def inception_module(x, filters, kernel_sizes=[1, 3, 5], prefix="inception"):
    """
    Create an inception module with multiple parallel convolutional paths

    Args:
        x: Input tensor
        filters: Number of filters per path
        kernel_sizes: List of kernel sizes for each path
        prefix: Prefix for layer names

    Returns:
        Tensor: Output of inception module
    """
    # Create parallel paths with different kernel sizes
    paths = []

    for i, kernel_size in enumerate(kernel_sizes):
        conv_name = f"{prefix}_conv{kernel_size}_{i}"
        path = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=l1_l2(l1=CONFIG["l1_reg"], l2=CONFIG["l2_reg"]),
            name=conv_name,
        )(x)

        if CONFIG["use_batch_norm"]:
            path = BatchNormalization(name=f"{conv_name}_bn")(path)

        paths.append(path)

    # Concatenate all paths
    return Concatenate(name=f"{prefix}_concat")(paths)


def residual_block(x, filters, kernel_size=3, dilation_rate=1, prefix="residual"):
    """
    Create a residual block with skip connection

    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size for convolution
        dilation_rate: Dilation rate for convolution
        prefix: Prefix for layer names

    Returns:
        Tensor: Output of residual block
    """
    # Skip connection
    residual = x

    # Check if dimensions match for skip connection
    if residual.shape[-1] != filters:
        residual = Conv1D(
            filters=filters, kernel_size=1, padding="same", name=f"{prefix}_shortcut"
        )(residual)

    # First convolution layer
    y = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=l1_l2(l1=CONFIG["l1_reg"], l2=CONFIG["l2_reg"]),
        name=f"{prefix}_conv1",
    )(x)

    if CONFIG["use_batch_norm"]:
        y = BatchNormalization(name=f"{prefix}_bn1")(y)
    elif CONFIG["use_layer_norm"]:
        y = LayerNormalization(name=f"{prefix}_ln1")(y)

    y = Activation("relu", name=f"{prefix}_relu1")(y)

    # Second convolution layer
    y = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_regularizer=l1_l2(l1=CONFIG["l1_reg"], l2=CONFIG["l2_reg"]),
        name=f"{prefix}_conv2",
    )(y)

    if CONFIG["use_batch_norm"]:
        y = BatchNormalization(name=f"{prefix}_bn2")(y)
    elif CONFIG["use_layer_norm"]:
        y = LayerNormalization(name=f"{prefix}_ln2")(y)

    # Add skip connection
    y = Add(name=f"{prefix}_add")([y, residual])
    y = Activation("relu", name=f"{prefix}_relu2")(y)

    return y


def build_cnn_model(input_shape, config=CONFIG):
    """
    Build enhanced CNN model with advanced features

    Args:
        input_shape (tuple): Shape of input data (lookback, features)
        config (dict): Model configuration parameters

    Returns:
        tensorflow.keras.Model: Compiled CNN model
    """
    # Input layer
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    # Initial convolution
    x = Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=l1_l2(l1=config["l1_reg"], l2=config["l2_reg"]),
        name="initial_conv",
    )(x)

    # Apply spatial dropout if configured
    if config["spatial_dropout"] > 0:
        x = SpatialDropout1D(config["spatial_dropout"], name="spatial_dropout_1")(x)

    # Add inception modules or residual blocks based on configuration
    for i, filters in enumerate(config["filters"]):
        layer_name = f"layer_{i+1}"

        if config["use_inception"]:
            x = inception_module(
                x,
                filters=filters,
                kernel_sizes=config["kernel_sizes"],
                prefix=f"inception_{i+1}",
            )

        if config["use_residual"]:
            x = residual_block(
                x,
                filters=filters,
                kernel_size=config["kernel_sizes"][0],
                dilation_rate=config["dilation_rates"][
                    min(i, len(config["dilation_rates"]) - 1)
                ],
                prefix=f"residual_{i+1}",
            )

        # Add pooling after each block (except the last one)
        if i < len(config["filters"]) - 1:
            x = MaxPooling1D(pool_size=2, name=f"pool_{i+1}")(x)

    # Global pooling to reduce sequence length
    x = GlobalAveragePooling1D(name="global_avg_pooling")(x)

    # Add dense layers
    for i, units in enumerate(config["dense_units"]):
        x = Dense(
            units,
            activation="relu",
            kernel_regularizer=l1_l2(l1=config["l1_reg"], l2=config["l2_reg"]),
            name=f"dense_{i+1}",
        )(x)

        if config["dropout_rate"] > 0:
            x = Dropout(config["dropout_rate"], name=f"dropout_{i+1}")(x)

    # Output layer
    outputs = Dense(1, name="output")(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs, name=config["model_name"])

    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


def plot_training_history(history, output_dir):
    """Plot and save training metrics"""
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("CNN Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"])
    plt.plot(history.history["val_mae"])
    plt.title("CNN Model MAE")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cnn_training_history.png"))
    plt.close()


def train_cnn_model(
    data_path="data/processed/dataset.npz", output_dir="models/cnn", config=None
):
    """
    Train enhanced CNN model with the provided configuration

    Args:
        data_path (str): Path to preprocessed data
        output_dir (str): Directory to save model and results
        config (dict): Model configuration parameters

    Returns:
        tensorflow.keras.Model: Trained CNN model
        dict: Training history
    """
    # Start timing
    start_time = time.time()
    print(f"\n🚀 Starting {CONFIG['model_name']} training...")

    # Use provided config or default
    if config is None:
        config = CONFIG

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)
    print(
        f"Data loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples"
    )

    # Create model
    model = build_cnn_model(
        input_shape=(X_train.shape[1], X_train.shape[2]), config=config
    )
    model.summary()

    # Set up callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when plateauing
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(
                output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        ),
    ]

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Plot training history
    plot_training_history(history, output_dir)

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Test Loss: {test_loss:.4f}")
    print(f"📊 Test MAE: {test_mae:.4f}")

    # Save evaluation metrics
    eval_metrics = {
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "training_time_seconds": time.time() - start_time,
        "epochs_trained": len(history.history["loss"]),
        "final_learning_rate": float(
            history.history.get("lr", [config["learning_rate"]])[-1]
        ),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # Save final model
    model.save(os.path.join(output_dir, "final_model.keras"))

    # Training complete
    end_time = time.time()
    print(
        f"✅ {CONFIG['model_name']} training complete. Total time: {end_time - start_time:.2f} seconds"
    )

    return model, history


if __name__ == "__main__":
    train_cnn_model(data_path="data/processed/dataset.npz", output_dir="models/cnn")
