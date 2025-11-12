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
from helpers.model import build_model, evaluate_saved_model, evaluate_saved_qat_model

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



int8bo_top_cfg = {"num_dscnn_layers": 2, "first_conv_filters": 64, "first_conv_kernel": (8, 4), "first_conv_stride": (2, 2), "depthwise_kernel": (5, 5), "pointwise_filters": 96, "pooling_function": "max", "dropout_rate": 0.3}
rs_top_cfg = {"num_dscnn_layers": 2, "first_conv_filters": 64, "first_conv_kernel": (10, 4), "first_conv_stride": (2, 2), "depthwise_kernel": (5, 5), "pointwise_filters": 96, "pooling_function": "max", "dropout_rate": 0.0}
vabo_top_cfg = {"num_dscnn_layers": 6, "first_conv_filters": 64, "first_conv_kernel": (8, 4), "first_conv_stride": (2, 2), "depthwise_kernel": (3, 3), "pointwise_filters": 96, "pooling_function": "gap", "dropout_rate": 0.3}


model_config = {
   "best_rs_qat_dscnn.weights.h5": rs_top_cfg,
   "best_vabo_qat_dscnn.weights.h5": vabo_top_cfg,
   "best_int8bo_qat_dscnn.weights.h5": int8bo_top_cfg,
}

model_paths = [
    "./models/best_rs_dscnn.keras",
    "./models/rs/best_rs_qat_dscnn.weights.h5",
    "./models/best_vabo_dscnn.keras",
    "./models/vabo/best_vabo_qat_dscnn.weights.h5",
    "./models/best_int8bo_dscnn.keras",
    "./models/int8bo/best_int8bo_qat_dscnn.weights.h5"
]

results = []

for path in model_paths:
    if "qat" in path:
        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, int8_acc, int8_size, int8_latency = evaluate_saved_qat_model(
            path,
            build_model,  
            input_shape,
            num_classes,
            model_config[path.split("/")[-1]],
            X_train, y_train, X_val, y_val, X_test, y_test,
            optimizer=None,
        )
        
        results.append({
            "Model": path.split("/")[-1],
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Val Loss": val_loss,
            "Val Acc": val_acc,
            "Test Loss": test_loss,
            "Test Acc": test_acc,
            "INT8 Accuracy": int8_acc,
            "INT8 Latency": int8_latency,
            "INT8 Model Size": int8_size
        })
    else:
        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, int8_acc, int8_size, int8_latency = evaluate_saved_model(
            path, X_train, y_train, X_val, y_val, X_test, y_test
        )
        results.append({
            "Model": path.split("/")[-1],
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Val Loss": val_loss,
            "Val Acc": val_acc,
            "Test Loss": test_loss,
            "Test Acc": test_acc,
            "INT8 Accuracy": int8_acc,
            "INT8 Latency": int8_latency,
            "INT8 Model Size": int8_size
        })


df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Test Acc", ascending=False)

print(df_results)

df_results.to_csv("multi_stage_search_strategy_comparison.csv")