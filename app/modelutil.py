import os
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, Reshape, SpatialDropout3D,
    BatchNormalization, TimeDistributed, Flatten
)

def load_model() -> Sequential:
    # Get the base directory (assuming this file is in the same structure as app.py)
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"
    
    # Define checkpoint paths (try multiple possible extensions/names)
    checkpoint_path = None
    possible_names = [
        "checkpoint",
        "checkpoint.ckpt",
        "checkpoint.index",
        "model.ckpt",
        "model.h5"
    ]
    
    # Check for existing checkpoint files
    for name in possible_names:
        test_path = MODEL_DIR / name
        if test_path.exists():
            checkpoint_path = test_path
            break
    
    if checkpoint_path is None:
        available_files = "\n".join([f.name for f in MODEL_DIR.glob("*")])
        raise FileNotFoundError(
            f"No model checkpoint found in {MODEL_DIR}\n"
            f"Available files:\n{available_files}"
        )

    model = Sequential()
    
    # Model architecture (unchanged)
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

    try:
        # Try loading weights with different approaches
        if str(checkpoint_path).endswith('.h5'):
            model.load_weights(str(checkpoint_path))
        else:
            # For TensorFlow checkpoints
            model.load_weights(str(checkpoint_path).replace('.index', ''))
        
        return model
    except Exception as e:
        raise ValueError(f"Error loading weights from {checkpoint_path}: {str(e)}")
