import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    BatchNormalization,
)
from keras.api.regularizers import l1_l2
from keras.api.optimizers import Adam
from keras.api.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import load_data

# Model configuration
CONFIG = {
    "model_name": "enhanced_lstm",
    "lstm_units": [64, 32],
    "dense_units": [16],
    "dropout_rate": 0.2,
    "recurrent_dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "patience": 10,
    "use_bidirectional": True,
    "use_batch_norm": True,
    "use_attention": False,  # Removed attention layer
    "l1_reg": 0.0001,
    "l2_reg": 0.0001,
}


def build_lstm_model(input_shape, config=CONFIG):
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    for i, units in enumerate(config["lstm_units"]):
        return_sequences = i < len(config["lstm_units"]) - 1

        lstm_layer = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=config["dropout_rate"],
            recurrent_dropout=config["recurrent_dropout"],
            kernel_regularizer=l1_l2(config["l1_reg"], config["l2_reg"]),
            recurrent_regularizer=l1_l2(config["l1_reg"], config["l2_reg"]),
            name=f"lstm_{i+1}",
        )

        if config["use_bidirectional"]:
            x = Bidirectional(lstm_layer, name=f"bidirectional_{i+1}")(x)
        else:
            x = lstm_layer(x)

        if config["use_batch_norm"]:
            x = BatchNormalization(name=f"batch_norm_{i+1}")(x)

    for i, units in enumerate(config["dense_units"]):
        x = Dense(
            units,
            activation="relu",
            kernel_regularizer=l1_l2(config["l1_reg"], config["l2_reg"]),
            name=f"dense_{i+1}",
        )(x)
        x = Dropout(config["dropout_rate"], name=f"dropout_{i+1}")(x)

    outputs = Dense(1, name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=config["model_name"])
    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("LSTM Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Train")
    plt.plot(history.history["val_mae"], label="Validation")
    plt.title("LSTM Model MAE")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_training_history.png"))
    plt.close()


def train_lstm_model(
    data_path="data/processed/dataset.npz", output_dir="models/lstm", config=None
):
    start_time = time.time()
    print(f"\n🚀 Starting {CONFIG['model_name']} training...")

    if config is None:
        config = CONFIG

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)
    print(
        f"Data loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples"
    )

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), config)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(
                output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history, output_dir)

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Test Loss: {test_loss:.4f}")
    print(f"📊 Test MAE: {test_mae:.4f}")

    metrics = {
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
        json.dump(metrics, f, indent=2)

    model.save(os.path.join(output_dir, "final_model.keras"))

    print(
        f"✅ {CONFIG['model_name']} training complete. Total time: {time.time() - start_time:.2f} seconds"
    )
    return model, history


if __name__ == "__main__":
    train_lstm_model(data_path="data/processed/dataset.npz", output_dir="models/lstm")
