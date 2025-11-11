import math
import numpy as np
import random
import json
import gc
import time
import os

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")

from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot

SEED = 42  
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "../../data/processed/gsc_mfcc40_ds_cnn.npz"

data = np.load(DATA_PATH, allow_pickle=True)

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]
label_to_index = data["label_to_index"].item()

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

num_classes = len(label_to_index)
input_shape = X_train.shape[1:]

print("Length of Training Data", len(X_train))
print("Number of Clases", num_classes)
print("Shape of Input", input_shape)

interpreter = tf.lite.Interpreter(model_path="../b2rs-best-dscnn-model-int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']
print("Input quantization:", input_scale, input_zero_point)
print("Output quantization:", output_scale, output_zero_point)

correct = 0
total = len(X_test)

for i in range(total):
    sample = X_test[i:i+1].astype(np.float32)

    # Quantize input (from float32 → int8/uint8)
    if input_scale > 0:
        sample_q = sample / input_scale + input_zero_point
        sample_q = np.clip(sample_q, 0, 255).astype(np.uint8)
    else:
        sample_q = sample

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], sample_q)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output back to float
    if output_scale > 0:
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    pred_label = np.argmax(output_data)
    true_label = y_test[i]

    if pred_label == true_label:
        correct += 1

accuracy = correct / total
print(f"TFLite INT8 Model Accuracy: {accuracy * 100:.2f}%")
