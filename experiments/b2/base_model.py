import math
import numpy as np

import tensorflow as tf
# from tensorflow_model_optimization.python.core.keras.compat import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow_model_optimization as keras

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
    x = layers.DepthwiseConv2D(depthwise_kernel, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=pointwise_filters, kernel_size=(1,1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

def build_model(input_shape, num_classes, num_dscnn_layers):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=64, kernel_size=(10, 4), strides=(2, 2), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(num_dscnn_layers):
        x = build_dscnn_layer(x, depthwise_kernel=(3,3), pointwise_filters=(64))

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Keyword Spotting Model") 

    return model

print("=================Model 1 With 4 DSCNN layers================")
model1 = build_model(input_shape, num_classes, 4)
model1.summary()
print("=================Model 2 with 6 DSCNN Layers================")
model2 = build_model(input_shape, num_classes, 6)
model2.summary()








