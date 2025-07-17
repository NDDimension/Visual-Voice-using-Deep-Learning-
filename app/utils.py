import tensorflow as tf
from typing import List, Tuple
import cv2
import os
from pathlib import Path

# Initialize Path objects for directory structure
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to project_root/
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "s1"
ALIGNMENTS_DIR = DATA_DIR / "alignments" / "s1"

# Vocabulary setup (unchanged)
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str) -> tf.Tensor:
    """Load and preprocess video frames"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame[190:236, 80:220, :])
    finally:
        cap.release()
    
    if not frames:
        raise ValueError(f"No frames extracted from {path}")
    
    frames = tf.cast(frames, tf.float32)
    return (frames - tf.math.reduce_mean(frames)) / tf.math.reduce_std(frames)

def load_alignments(path: str) -> tf.Tensor:
    """Load and process alignment files"""
    try:
        with open(path, "r") as f:
            lines = [line.split() for line in f.readlines()]
        
        tokens = [" " + line[2] for line in lines if line[2] != "sil"]
        chars = tf.strings.unicode_split("".join(tokens), "UTF-8")
        return char_to_num(chars)[1:]  # Skip the first token (usually empty)
    except Exception as e:
        raise ValueError(f"Error processing {path}: {str(e)}")

def load_data(path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Main data loading function with proper path handling"""
    path = bytes.decode(path.numpy()) if isinstance(path, tf.Tensor) else path
    file_name = Path(path).stem
    
    # Construct absolute paths
    video_path = str(VIDEO_DIR / f"{file_name}.mpg")
    alignment_path = str(ALIGNMENTS_DIR / f"{file_name}.align")
    
    return load_video(video_path), load_alignments(alignment_path)

def get_available_videos() -> List[str]:
    """Get list of available video names (without extension)"""
    return sorted(f.stem for f in VIDEO_DIR.glob("*.mpg") if f.is_file())
