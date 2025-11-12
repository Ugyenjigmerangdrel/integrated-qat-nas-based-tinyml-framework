import math
import numpy as np
import random
import json
import gc
import time
import os
import pandas as pd

import tensorflow as tf
from helpers.initialize_gpu import initialize_gpu

gpu_status = initialize_gpu()
print(gpu_status)

from tensorflow_model_optimization.python.core.keras.compat import keras

from helpers.data_loader import load_data
from helpers.model import evaluate_saved_model

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "../../data/processed/gsc_mfcc40_ds_cnn.npz"

X_train, y_train, X_val, y_val, X_test, y_test, label_to_index, num_classes, input_shape = load_data(DATA_PATH)

print("Length of Training Data", len(X_train))
print("Number of Clases", num_classes)
print("Shape of Input", input_shape)

model_paths = [
    "./models/best_rs_dscnn.keras",
    # "./models/best_rs_qat_dscnn.keras",
    "./models/best_vabo_dscnn.keras",
    # "./models/best_vabo_qat_dscnn.keras",
    "./models/best_int8bo_dscnn.keras",
    # "./models/best_int8bo_qat_dscnn.keras"
]

results = []

for path in model_paths:
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = evaluate_saved_model(
        path, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results.append({
        "Model": path.split("/")[-1],
        "Train Loss": train_loss,
        "Train Acc": train_acc,
        "Val Loss": val_loss,
        "Val Acc": val_acc,
        "Test Loss": test_loss,
        "Test Acc": test_acc
    })


df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Test Acc", ascending=False)

print(df_results)

df_results.to_csv("multi_stage_search_strategy_comparison.csv")