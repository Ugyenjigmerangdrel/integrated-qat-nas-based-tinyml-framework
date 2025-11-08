#!/usr/bin/env python
"""
preprocessing_b0.py

Prepare Google Speech Commands dataset for DS-CNN training.

- Uses Google Speech Commands v0.01/v0.02 layout:
    DATA_ROOT/
        yes/
        no/
        ...
        _background_noise_/
        validation_list.txt
        testing_list.txt

- 10 target keywords:
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"

- All other 20 words are merged into a single "unknown" class.

- Adds a "silence" class using zeros (easy baseline).
  (You can later switch to background-noise based silence if you like.)

- Extracts 40 MFCCs with:
    frame length = 40 ms (n_fft=640)
    stride       = 20 ms (hop_length=320)
  -> 49 frames per 1 second, so final feature is (49, 40).

- Splits:
    Train = all files not in validation_list / testing_list
    Val   = validation_list.txt
    Test  = testing_list.txt

Saves:
    .npz file with X_train, y_train, X_val, y_val, X_test, y_test, label_to_index
"""

import os
import numpy as np
import librosa
from tqdm import tqdm
import random

# =========================================================
# CONFIG: adjust these paths for your project
# =========================================================
DATA_ROOT = "../../data/raw/"  # <-- change if needed
OUTPUT_DIR = "../../data/processed"              # <-- change if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gsc_mfcc40_ds_cnn.npz")

SAMPLE_RATE = 16000
DURATION = 1.0               # seconds
N_MFCC = 40
N_FFT = 640                  # 40 ms window
HOP_LENGTH = 320             # 20 ms stride
TARGET_FRAMES = 49           # we want 49 frames -> 49 x 40 features

TARGET_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# Helpers to load split lists
# =========================================================
def load_list(path):
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


VAL_LIST_PATH = os.path.join(DATA_ROOT, "validation_list.txt")
TEST_LIST_PATH = os.path.join(DATA_ROOT, "testing_list.txt")

val_list = load_list(VAL_LIST_PATH)
test_list = load_list(TEST_LIST_PATH)


def iter_all_wavs(root):
    """Yield relative paths like 'yes/xxx.wav' for all label folders except background noise."""
    for label in sorted(os.listdir(root)):
        full_dir = os.path.join(root, label)
        if not os.path.isdir(full_dir):
            continue
        if label == "_background_noise_":
            continue
        for fname in os.listdir(full_dir):
            if fname.endswith(".wav"):
                rel_path = os.path.join(label, fname)
                # Convert backslashes to forward slashes in case of Windows-style paths
                rel_path = rel_path.replace("\\", "/")
                yield rel_path


all_wavs = list(iter_all_wavs(DATA_ROOT))

# Build file sets for train/val/test using provided lists
val_files = [p for p in all_wavs if p in val_list]
test_files = [p for p in all_wavs if p in test_list]
train_files = [p for p in all_wavs if p not in val_list and p not in test_list]

print(f"Total wavs: {len(all_wavs)}")
print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


# =========================================================
# MFCC extraction
# =========================================================
def extract_mfcc_from_wave(y):
    """
    Take a 1D waveform y (already at SAMPLE_RATE, already 1s padded/truncated),
    and return MFCC of shape (49, 40).
    """
    # Compute MFCCs.
    # center=False so we don't pad at the edges; this should give exactly 49 frames,
    # but we enforce it anyway with pad/crop just in case.
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )
    mfcc = mfcc.T  # shape (time, n_mfcc)

    # Enforce fixed number of frames
    if mfcc.shape[0] < TARGET_FRAMES:
        pad_len = TARGET_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_len), (0, 0)), mode="constant")
    elif mfcc.shape[0] > TARGET_FRAMES:
        mfcc = mfcc[:TARGET_FRAMES, :]

    assert mfcc.shape == (TARGET_FRAMES, N_MFCC), f"Got {mfcc.shape}"
    return mfcc.astype(np.float32)


def extract_mfcc(file_path):
    """
    Load audio from file_path, force to 1 second (16000 samples),
    then call extract_mfcc_from_wave.
    """
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    # Force exactly 1 second
    if len(y) < SAMPLE_RATE:
        y = np.pad(y, (0, SAMPLE_RATE - len(y)))
    else:
        y = y[:SAMPLE_RATE]

    return extract_mfcc_from_wave(y)


# =========================================================
# Build features for a split
# =========================================================
def map_label(folder_name):
    """
    Map original folder name to our final class label.
    - 10 target words keep their own label
    - everything else becomes UNKNOWN_LABEL
    """
    if folder_name in TARGET_WORDS:
        return folder_name
    else:
        return UNKNOWN_LABEL


def build_split(file_list, split_name):
    X_list = []
    y_list = []

    for rel_path in tqdm(file_list, desc=f"Building {split_name}"):
        full_path = os.path.join(DATA_ROOT, rel_path)
        label_folder = rel_path.split("/")[0]
        label = map_label(label_folder)

        mfcc = extract_mfcc(full_path)
        X_list.append(mfcc)
        y_list.append(label)

    X = np.stack(X_list).astype(np.float32)  # shape (N, 49, 40)
    y = np.array(y_list)
    return X, y


def add_silence_examples(X, y, split_name, fraction=0.1):
    """
    Add synthetic 'silence' examples as zero MFCCs.
    fraction = proportion of silence relative to existing examples.
    """
    num_existing = X.shape[0]
    num_silence = int(num_existing * fraction)
    if num_silence == 0:
        return X, y

    silence_mfcc = np.zeros((TARGET_FRAMES, N_MFCC), dtype=np.float32)
    silence_stack = np.repeat(silence_mfcc[np.newaxis, ...], num_silence, axis=0)
    silence_labels = np.array([SILENCE_LABEL] * num_silence)

    X_new = np.concatenate([X, silence_stack], axis=0)
    y_new = np.concatenate([y, silence_labels], axis=0)

    print(f"[{split_name}] Added {num_silence} silence examples.")
    return X_new, y_new


def main():
    # Build train/val/test splits
    X_train, y_train = build_split(train_files, "train")
    X_val, y_val = build_split(val_files, "val")
    X_test, y_test = build_split(test_files, "test")

    # Add silence class to each split (10% of size as a simple baseline)
    X_train, y_train = add_silence_examples(X_train, y_train, "train", fraction=0.1)
    X_val, y_val = add_silence_examples(X_val, y_val, "val", fraction=0.1)
    X_test, y_test = add_silence_examples(X_test, y_test, "test", fraction=0.1)

    # Build label map (consistent order: 10 keywords + unknown + silence)
    all_labels = TARGET_WORDS + [UNKNOWN_LABEL, SILENCE_LABEL]
    label_to_index = {lab: idx for idx, lab in enumerate(all_labels)}
    print("Label to index:", label_to_index)

    # Convert string labels to integer indices
    y_train_idx = np.array([label_to_index[l] for l in y_train], dtype=np.int64)
    y_val_idx = np.array([label_to_index[l] for l in y_val], dtype=np.int64)
    y_test_idx = np.array([label_to_index[l] for l in y_test], dtype=np.int64)

    # Optional: normalisation (global mean/std over training set)
    # Many KWS setups normalise features; you can comment out if you don't want it.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    np.savez_compressed(
        OUTPUT_PATH,
        X_train=X_train_norm,
        y_train=y_train_idx,
        X_val=X_val_norm,
        y_val=y_val_idx,
        X_test=X_test_norm,
        y_test=y_test_idx,
        label_to_index=label_to_index,
        mean=mean,
        std=std,
    )

    print("Saved processed dataset to:", OUTPUT_PATH)
    print("Shapes:")
    print("  X_train:", X_train_norm.shape, "y_train:", y_train_idx.shape)
    print("  X_val:", X_val_norm.shape, "y_val:", y_val_idx.shape)
    print("  X_test:", X_test_norm.shape, "y_test:", y_test_idx.shape)


if __name__ == "__main__":
    main()
