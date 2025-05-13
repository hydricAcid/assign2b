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
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    GlobalAveragePooling1D,
    Masking,
)
from keras.api.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    LearningRateScheduler,
)
from keras.api.optimizers import Adam
from keras.api.regularizers import l1_l2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_data

# Configuration settings
CONFIG = {
    "model_name": "enhanced_gru",
    "gru_units": [96, 64],  # Units in each GRU layer
    "dense_units": [32, 16],  # Units in each Dense layer
    "dropout_rate": 0.3,
    "recurrent_dropout": 0.2,
    "learning_rate": 0.002,
    "batch_size": 64,
    "epochs": 10,
    "patience": 10,
    "use_residual": True,
    "use_multi_head_attention": True,
    "attention_heads": 4,
    "use_batch_norm": False,
    "use_layer_norm": True,
    "l1_reg": 0.0001,
    "l2_reg": 0.0001,
    "schedule_learning_rate": True,
}


def lr_schedule(epoch, lr):
    """Custom learning rate schedule with warm-up and decay"""
    if epoch < 5:  # Warm up for 5 epochs
        return lr * (epoch + 1) / 5
    else:
        # Decay learning rate exponentially after warm-up
        decay_rate = 0.95
        return lr * (decay_rate ** (epoch - 5))


def build_gru_model(input_shape, config=CONFIG):
    """
    Build enhanced GRU model with advanced features

    Args:
        input_shape (tuple): Shape of input data (lookback, features)
        config (dict): Model configuration parameters

    Returns:
        tensorflow.keras.Model: Compiled GRU model
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # Add masking to handle variable length sequences or missing values
    x = Masking(mask_value=0.0)(inputs)

    # Track residual input for skip connections
    residual = x

    # Add GRU layers
    for i, units in enumerate(config["gru_units"]):
        return_sequences = (
            i < len(config["gru_units"]) - 1 or config["use_multi_head_attention"]
        )

        # Apply GRU layer with regularization
        gru_layer = GRU(
            units,
            return_sequences=return_sequences,
            dropout=config["dropout_rate"],
            recurrent_dropout=config["recurrent_dropout"],
            kernel_regularizer=l1_l2(l1=config["l1_reg"], l2=config["l2_reg"]),
            recurrent_regularizer=l1_l2(l1=config["l1_reg"], l2=config["l2_reg"]),
        )

        x = gru_layer(x)

        # Apply normalization
        if config["use_batch_norm"]:
            x = BatchNormalization()(x)
        elif config["use_layer_norm"]:
            x = LayerNormalization()(x)

        # Apply residual connection if configured and shapes match
        if config["use_residual"] and i > 0 and return_sequences:
            # Project residual to match shape if needed
            if residual.shape[-1] != x.shape[-1]:
                residual = Dense(
                    units,
                    kernel_regularizer=l1_l2(l1=config["l1_reg"], l2=config["l2_reg"]),
                )(residual)
            x = Add()([x, residual])

        # Update residual
        residual = x

    # Apply multi-head attention if configured
    if config["use_multi_head_attention"]:
        # Self-attention mechanism
        attention_output = MultiHeadAttention(
            num_heads=config["attention_heads"],
            key_dim=config["gru_units"][-1] // config["attention_heads"],
        )(x, x)

        # Add & normalize (like in Transformer architecture)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)

        # Pool over time dimension
        x = GlobalAveragePooling1D()(x)

    # Add dense layers
    for units in config["dense_units"]:
        x = Dense(
            units,
            activation="relu",
            kernel_regularizer=l1_l2(l1=config["l1_reg"], l2=config["l2_reg"]),
        )(x)
        x = Dropout(config["dropout_rate"])(x)

    # Output layer
    outputs = Dense(1)(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)

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
    plt.title("GRU Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"])
    plt.plot(history.history["val_mae"])
    plt.title("GRU Model MAE")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gru_training_history.png"))
    plt.close()


def train_gru_model(
    data_path="data/processed/dataset.npz", output_dir="models/gru", config=None
):
    """
    Train enhanced GRU model with the provided configuration

    Args:
        data_path (str): Path to preprocessed data
        output_dir (str): Directory to save model and results
        config (dict): Model configuration parameters

    Returns:
        tensorflow.keras.Model: Trained GRU model
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
    model = build_gru_model(
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

    # Add learning rate scheduler if configured
    if config["schedule_learning_rate"]:
        callbacks.append(LearningRateScheduler(lr_schedule, verbose=1))

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
    train_gru_model(data_path="data/processed/dataset.npz", output_dir="models/gru")
