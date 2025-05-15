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
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    BatchNormalization,
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
    "model_name": "lstm",
    "lstm_units": [64, 32],
    "dense_units": [16],
    "dropout_rate": 0.3,
    "recurrent_dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "patience": 10,
    "use_bidirectional": True,
    "use_batch_norm": True,
    "l1_reg": 0.0001,
    "l2_reg": 0.0001,
}


def build_lstm_model(input_shape, config=CONFIG):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    for i, units in enumerate(config["lstm_units"]):
        return_seq = i < len(config["lstm_units"]) - 1
        lstm = LSTM(
            units,
            return_sequences=return_seq,
            dropout=config["dropout_rate"],
            recurrent_dropout=config["recurrent_dropout"],
            kernel_regularizer=l1_l2(config["l1_reg"], config["l2_reg"]),
            recurrent_regularizer=l1_l2(config["l1_reg"], config["l2_reg"]),
        )
        x = Bidirectional(lstm)(x) if config["use_bidirectional"] else lstm(x)
        if config["use_batch_norm"]:
            x = BatchNormalization()(x)
    for i, units in enumerate(config["dense_units"]):
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
    plt.savefig(os.path.join(out_dir, "lstm_training_history.png"))
    plt.close()


def train_lstm_model():
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"models/lstm_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        "data/processed/dataset.npz"
    )
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
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
            f"[{timestamp}] LSTM trained. MAE: {test_mae:.4f} | Saved to: {out_dir}\n"
        )
    print(f"✅ LSTM model trained. Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    train_lstm_model()
