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


def build_dscnn_layer(x, depthwise_kernel=(3,3), pointwise_filters=64):
    x = keras.layers.DepthwiseConv2D(depthwise_kernel, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=pointwise_filters, kernel_size=(1,1), padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x

def pooling_layer(x, selected_pooling):
    if selected_pooling=="gap":
        x = keras.layers.GlobalAveragePooling2D()(x)
    elif selected_pooling=="max":
        x = keras.layers.GlobalMaxPooling2D()(x)

    return x

def build_model(input_shape, num_classes, cfg):
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=cfg["first_conv_filters"], kernel_size=cfg["first_conv_kernel"], strides=cfg["first_conv_stride"], padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    for _ in range(cfg["num_dscnn_layers"]):
        x = build_dscnn_layer(x, depthwise_kernel=cfg["depthwise_kernel"], pointwise_filters=cfg["pointwise_filters"])

    if cfg["dropout_rate"] > 0:
        x = keras.layers.Dropout(cfg["dropout_rate"])(x)

    x = pooling_layer(x, cfg["pooling_function"])
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs, name="keyword-spotting-model") 

    return model


# config reuse from the b2 - random search best model
# "5": [0.19360148906707764, 0.942271888256073, 0.08052890300750733, 157.765625, 70.42168712615967, {"num_dscnn_layers": 2, "first_conv_filters": 64, "first_conv_kernel": [10, 4], "first_conv_stride": [2, 2], "depthwise_kernel": [5, 5], "pointwise_filters": 96, "pooling_function": "max", "dropout_rate": 0.0}]
cfg = {"num_dscnn_layers": 2, "first_conv_filters": 64, "first_conv_kernel": (10, 4), "first_conv_stride": (2, 2), "depthwise_kernel": (5, 5), "pointwise_filters": 96, "pooling_function": "max", "dropout_rate": 0.0}

model = build_model(input_shape, num_classes, cfg)

initial_lr = 5e-4
opt = keras.optimizers.Adam(learning_rate=initial_lr)

model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.load_weights("b2rs-best-dscnn-model.weights.h5")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy before Quantization: {test_acc * 100:.2f}%")

#PTQ Layer
model.save("b2rs-best-dscnn-model.keras")

