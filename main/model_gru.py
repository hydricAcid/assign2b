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
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Add,
    Masking,
    MultiHeadAttention,
    GlobalAveragePooling1D,
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

from utils.preprocess import load_data

CONFIG = {
    "model_name": "gru",
    "gru_units": [64, 32],
    "dense_units": [32],
    "dropout_rate": 0.3,
    "recurrent_dropout": 0.1,
    "learning_rate": 0.0008,
    "batch_size": 64,
    "epochs": 40,
    "patience": 15,
    "use_residual": True,
    "use_multi_head_attention": False,
    "attention_heads": 4,
    "use_batch_norm": False,
    "use_layer_norm": True,
    "l1_reg": 0.0001,
    "l2_reg": 0.0001,
    "schedule_learning_rate": True,
}


def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr * (epoch + 1) / 5
    else:
        decay_rate = 0.95
        return lr * (decay_rate ** (epoch - 5))


def build_gru_model(input_shape, config=CONFIG):
    inputs = Input(shape=input_shape)
    x = Masking(mask_value=0.0)(inputs)
    residual = x

    for i, units in enumerate(config["gru_units"]):
        return_sequences = (
            i < len(config["gru_units"]) - 1 or config["use_multi_head_attention"]
        )
        x = GRU(
            units,
            return_sequences=return_sequences,
            dropout=config["dropout_rate"],
            recurrent_dropout=config["recurrent_dropout"],
            kernel_regularizer=l1_l2(config["l1_reg"], config["l2_reg"]),
            recurrent_regularizer=l1_l2(config["l1_reg"], config["l2_reg"]),
        )(x)

        if config["use_batch_norm"]:
            x = BatchNormalization()(x)
        elif config["use_layer_norm"]:
            x = LayerNormalization()(x)

        if config["use_residual"] and return_sequences:
            if residual.shape[-1] != x.shape[-1]:
                residual = Dense(units)(residual)
            x = Add()([x, residual])
        residual = x

    if config["use_multi_head_attention"]:
        attention_output = MultiHeadAttention(
            num_heads=config["attention_heads"],
            key_dim=config["gru_units"][-1] // config["attention_heads"],
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)

    for units in config["dense_units"]:
        x = Dense(units, activation="relu")(x)
        x = Dropout(config["dropout_rate"])(x)

    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(config["learning_rate"]), loss="mse", metrics=["mae"])
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
    plt.savefig(os.path.join(out_dir, "gru_training_history.png"))
    plt.close()


def train_gru_model():
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"models/gru_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        "data/processed/dataset.npz"
    )

    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [
        ModelCheckpoint(
            os.path.join(out_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
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

    if CONFIG["schedule_learning_rate"]:
        callbacks.append(LearningRateScheduler(lr_schedule, verbose=1))

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
            f"[{timestamp}] GRU trained. MAE: {test_mae:.4f} | Saved to: {out_dir}\n"
        )
    print(f"✅ GRU model trained. Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    train_gru_model()
