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

print("Top Random Search Derived Model Config Evaluation \n")
print(evaluate_saved_model("./models/best_rs_dscnn.keras",X_train, y_train, X_val, y_val, X_test, y_test))
