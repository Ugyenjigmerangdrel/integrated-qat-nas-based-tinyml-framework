import os
import librosa
import numpy as np

# ==== Paths to dataset and split lists ====
DATA_PATH = "../../data/raw/"  # root folder of the raw Speech Commands audio

TRAIN_LIST = "../data-partition/train_files.txt"      # e.g. "train_list.txt"
VAL_LIST = "../data-partition/val_files.txt"        # e.g. "validation_list.txt"
TEST_LIST = "../data-partition/test_files.txt"       # e.g. "testing_list.txt"

# save the processed numpy files for future consistant use case
PROCESSED_ROOT = "../../data/processed_40/" 

# ==== Hyperparameters ====
SAMPLE_RATE = 16000
DURATION = 1.0
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160


def load_audio(file_path, target_sr=SAMPLE_RATE, duration=DURATION):
    """Load and preprocess audio file"""
    audio, sr = librosa.load(file_path, sr=target_sr, duration=duration)
    # Pad or trim to exactly duration * sample_rate samples
    audio = librosa.util.fix_length(audio, size=int(target_sr * duration))
    return audio


def extract_features(file_path):
    audio = load_audio(file_path)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def read_list(list_path):
    """Read a txt file of relative paths, one per line."""
    with open(list_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def build_label_map(data_path):
    """
    Build a label -> index map from folder names
    (ignoring _background_noise_).
    """
    all_items = sorted(os.listdir(data_path))
    classes = [
        item for item in all_items
        if os.path.isdir(os.path.join(data_path, item))
        and item != "_background_noise_"
    ]
    label_map = {label: idx for idx, label in enumerate(classes)}
    return label_map


def create_split(file_list, data_path, label_map, split_name, save_root=None):
    """
    Given a list of relative paths (like 'yes/0a7c2a8d_nohash_0.wav'),
    extract features and labels, and optionally save each mel as a .npy file.

    split_name: "train", "val", or "test"
    save_root: root folder where processed files will be saved
    """
    X, y = [], []

    for rel_path in file_list:
        # First part of the path is the label (folder name)
        label = rel_path.split("/")[0]
        if label not in label_map:
            # Skip any files with labels we didn't include
            continue

        full_path = os.path.join(data_path, rel_path)
        if not os.path.isfile(full_path):
            # In case a path in the txt file doesn't exist
            continue

        # Extract mel-spectrogram
        features = extract_features(full_path)
        X.append(features)
        y.append(label_map[label])

        # Save to .npy with same relative path (but .npy extension)
        if save_root is not None:
            # Example: processed_mels/train/yes/0a7c2a8d_nohash_0.npy
            base_rel, _ = os.path.splitext(rel_path)
            out_path = os.path.join(save_root, split_name, base_rel + ".npy")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, features.astype(np.float32))

    return np.array(X), np.array(y)


# ============ MAIN PIPELINE ============

# 1. Build label map from folders
label_map = build_label_map(DATA_PATH)

# 2. Read file lists
train_files = read_list(TRAIN_LIST)
val_files = read_list(VAL_LIST)
test_files = read_list(TEST_LIST)

# 3. Create splits and save processed .npy files
X_train, y_train = create_split(
    train_files, DATA_PATH, label_map,
    split_name="train", save_root=PROCESSED_ROOT
)

X_val, y_val = create_split(
    val_files, DATA_PATH, label_map,
    split_name="val", save_root=PROCESSED_ROOT
)

X_test, y_test = create_split(
    test_files, DATA_PATH, label_map,
    split_name="test", save_root=PROCESSED_ROOT
)

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)
print("Label map:", label_map)
