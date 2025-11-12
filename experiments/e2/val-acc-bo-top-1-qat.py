import math
import numpy as np
import random
import json
import gc
import time
import os

import tensorflow as tf
from helpers.initialize_gpu import initialize_gpu

gpu_status = initialize_gpu()
print(gpu_status)

from tensorflow_model_optimization.python.core.keras.compat import keras

from helpers.data_loader import load_data
from helpers.model import build_model, train_qat_model

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


vabo_top_cfg = {"num_dscnn_layers": 6, "first_conv_filters": 64, "first_conv_kernel": (8, 4), "first_conv_stride": (2, 2), "depthwise_kernel": (3, 3), "pointwise_filters": 96, "pooling_function": "gap", "dropout_rate": 0.3}

model = build_model(input_shape, num_classes, vabo_top_cfg)

initial_lr = 5e-4
opt = keras.optimizers.Adam(learning_rate=initial_lr)

train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = train_qat_model(model, opt, X_train, y_train, X_val, y_val, X_test, y_test, "./models/vabo/best_vabo_qat_dscnn.weights.h5", "./models/best_vabo_qat_dscnn.keras")

print(train_acc, val_acc, test_acc)






