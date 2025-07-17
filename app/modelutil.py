import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    MaxPool3D,
    Activation,
    TimeDistributed,
    Flatten,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(BASE_DIR, "..", "models", "checkpoint")

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

    model.add(Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer="he_normal", activation="softmax"))

    # ✅ Restore weights using tf.train.Checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.expect_partial()  # suppress optimizer-related warnings

    return model
