import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Add,
    Concatenate,
    SpatialDropout1D,
)
from keras.api.optimizers import Adam
from keras.api.regularizers import l1_l2
from keras.api.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from utils.preprocess import load_data

CONFIG = {
    "model_name": "cnn",
    "filters": [64, 128],
    "kernel_sizes": [3, 5],
    "dense_units": [64],
    "dropout_rate": 0.3,
    "spatial_dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 40,
    "patience": 10,
    "use_residual": True,
    "use_inception": True,
    "use_batch_norm": True,
    "use_layer_norm": False,
    "l1_reg": 0.0001,
    "l2_reg": 0.0001,
}


def inception_module(x, filters, kernel_sizes=[1, 3, 5], prefix="inception"):
    paths = []
    for i, k in enumerate(kernel_sizes):
        path = Conv1D(
            filters,
            k,
            padding="same",
            activation="relu",
            kernel_regularizer=l1_l2(CONFIG["l1_reg"], CONFIG["l2_reg"]),
            name=f"{prefix}_conv{k}_{i}",
        )(x)
        if CONFIG["use_batch_norm"]:
            path = BatchNormalization(name=f"{prefix}_bn{k}_{i}")(path)
        paths.append(path)
    return Concatenate(name=f"{prefix}_concat")(paths)


def residual_block(x, filters, kernel_size=3, dilation_rate=1, prefix="residual"):
    residual = x
    if residual.shape[-1] != filters:
        residual = Conv1D(filters, 1, padding="same", name=f"{prefix}_shortcut")(
            residual
        )
    y = Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=l1_l2(CONFIG["l1_reg"], CONFIG["l2_reg"]),
        name=f"{prefix}_conv1",
    )(x)
    if CONFIG["use_batch_norm"]:
        y = BatchNormalization(name=f"{prefix}_bn1")(y)
    elif CONFIG["use_layer_norm"]:
        y = LayerNormalization(name=f"{prefix}_ln1")(y)
    y = tf.keras.activations.relu(y)

    y = Conv1D(
        filters,
        kernel_size,
        padding="same",
        kernel_regularizer=l1_l2(CONFIG["l1_reg"], CONFIG["l2_reg"]),
        name=f"{prefix}_conv2",
    )(y)
    if CONFIG["use_batch_norm"]:
        y = BatchNormalization(name=f"{prefix}_bn2")(y)
    elif CONFIG["use_layer_norm"]:
        y = LayerNormalization(name=f"{prefix}_ln2")(y)

    y = Add(name=f"{prefix}_add")([y, residual])
    return tf.keras.activations.relu(y)


def build_cnn_model(input_shape, config=CONFIG):
    inputs = Input(shape=input_shape, name="input")
    x = Conv1D(32, 3, padding="same", activation="relu", name="initial_conv")(inputs)

    if config["spatial_dropout"] > 0:
        x = SpatialDropout1D(config["spatial_dropout"], name="spatial_dropout")(x)

    for i, filters in enumerate(config["filters"]):
        if config["use_inception"]:
            x = inception_module(
                x, filters, config["kernel_sizes"], prefix=f"inception_{i+1}"
            )
        if config["use_residual"]:
            x = residual_block(
                x, filters, config["kernel_sizes"][0], prefix=f"residual_{i+1}"
            )
        if i < len(config["filters"]) - 1:
            x = MaxPooling1D(pool_size=2, name=f"pool_{i+1}")(x)

    x = GlobalAveragePooling1D(name="global_avg_pool")(x)

    for i, units in enumerate(config["dense_units"]):
        x = Dense(
            units,
            activation="relu",
            kernel_regularizer=l1_l2(config["l1_reg"], config["l2_reg"]),
            name=f"dense_{i+1}",
        )(x)
        if config["dropout_rate"] > 0:
            x = Dropout(config["dropout_rate"], name=f"dropout_{i+1}")(x)

    outputs = Dense(1, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name=config["model_name"])
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def plot_history(history, out_dir):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="train")
    plt.plot(history.history["val_mae"], label="val")
    plt.title("MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cnn_training_history.png"))
    plt.close()


def train_cnn_model():
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"models/cnn_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        "data/processed/dataset.npz"
    )
    model = build_cnn_model((X_train.shape[1], X_train.shape[2]), CONFIG)
    callbacks = [
        ModelCheckpoint(
            os.path.join(out_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        EarlyStopping(
            monitor="val_loss", patience=CONFIG["patience"], restore_best_weights=True
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        TensorBoard(
            log_dir=os.path.join(
                out_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks,
    )
    plot_history(history, out_dir)
    model.save(os.path.join(out_dir, "final_model.keras"))
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(
            {"test_loss": float(test_loss), "test_mae": float(test_mae)}, f, indent=2
        )
    with open("train_log.txt", "a") as f:
        f.write(
            f"[{timestamp}] CNN trained. MAE: {test_mae:.4f} | Saved to: {out_dir}\n"
        )
    print(f"✅ CNN model trained. Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    train_cnn_model()
