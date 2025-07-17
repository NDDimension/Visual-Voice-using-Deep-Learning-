import tensorflow as tf
from typing import List
import cv2
import os

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path: str) -> List[float]:
    # print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> List[str]:
    # print(path)
    with open(path, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != "sil":
            tokens = [*tokens, " ", line[2]]
    return char_to_num(
        tf.reshape(tf.strings.unicode_split(tokens, input_encoding="UTF-8"), (-1))
    )[1:]


def load_data(path: str):
    path = bytes.decode(path.numpy())  # Convert from tensor to string
    print(f"Debug: Received path: {path}")

    # Get only the filename without any folder
    file_name = os.path.basename(path).split(".")[0]
    print(f"Debug: Extracted file_name: {file_name}")

    video_path = os.path.join("..", "data", "s1", f"{file_name}.mpg")
    alignment_path = os.path.join("..", "data", "alignments", "s1", f"{file_name}.align")

    print(f"Debug: video_path = {video_path}")
    print(f"Debug: alignment_path = {alignment_path}")

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments

