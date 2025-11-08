"""
prepare_gsc_data.py
Preprocess Google Speech Commands dataset for Keyword Spotting.
Extracts 40 MFCCs per frame (40ms window, 20ms stride) -> 49 x 40 features per 1s clip.
"""

import os
import numpy as np
import librosa
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
DATA_PATH = "../../data/raw/"
PROCESSED_DIR = "./processed_data"
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
N_MFCC = 40
N_FFT = 640    # 40ms window
HOP_LENGTH = 320  # 20ms stride
TARGET_WORDS = ["yes","no","up","down","left","right","on","off","stop","go"]
UNKNOWN_FRACTION = 0.2  # use fraction of unknowns
RANDOM_SEED = 42
os.makedirs(PROCESSED_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

def extract_mfcc(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) < SAMPLE_RATE:
        y = np.pad(y, (0, SAMPLE_RATE - len(y)))
    else:
        y = y[:SAMPLE_RATE]
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    return mfcc.T  # shape (49,40)

def build_dataset():
    all_labels = os.listdir(DATA_PATH)
    X, y = [], []

    # collect target + silence + unknown
    for label in tqdm(all_labels):
        dir_path = os.path.join(DATA_PATH, label)
        if not os.path.isdir(dir_path):
            continue
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.wav')]

        if label in TARGET_WORDS:
            selected_files = files
            label_name = label
        elif label == '_background_noise_':
            continue
        else:
            # unknowns
            selected_files = random.sample(files, int(len(files) * UNKNOWN_FRACTION))
            label_name = 'unknown'

        for file in selected_files:
            mfcc = extract_mfcc(file)
            X.append(mfcc)
            y.append(label_name)

    # Add silence: generate synthetic silence examples
    num_silence = int(len(X) * 0.1)
    for _ in range(num_silence):
        silence = np.zeros((49, 40))
        X.append(silence)
        y.append('silence')

    label_to_index = {w: i for i, w in enumerate(sorted(set(y)))}
    X = np.array(X)
    y = np.array([label_to_index[label] for label in y])

    # Split (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED)

    np.savez(os.path.join(PROCESSED_DIR, "gsc_mfcc_40_features.npz"),
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             label_to_index=label_to_index)
    print("✅ Saved processed dataset to", PROCESSED_DIR)

if __name__ == "__main__":
    build_dataset()
