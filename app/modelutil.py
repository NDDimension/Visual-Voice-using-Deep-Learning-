from pathlib import Path
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    MaxPool3D,
    Activation,
    Reshape,
    SpatialDropout3D,
    BatchNormalization,
    TimeDistributed,
    Flatten,
)

def load_model() -> Sequential:
    # Get the base directory (assuming this file is in the same structure as app.py)
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"

def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(
        Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True))
    )
    model.add(Dropout(0.5))

    model.add(
        Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True))
    )
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer="he_normal", activation="softmax"))

    model.load_weights(MODEL_DIR / "checkpoint")

    return model

