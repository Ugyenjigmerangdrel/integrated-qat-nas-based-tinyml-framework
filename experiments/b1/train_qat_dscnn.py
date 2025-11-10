import math
import numpy as np
import tensorflow as tf


from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot

DATA_PATH = "../../data/processed/gsc_mfcc40_ds_cnn.npz"

data = np.load(DATA_PATH, allow_pickle=True)

X_train = data["X_train"]   # (N_train, 49, 40)
y_train = data["y_train"]   # (N_train,)
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

# validation of if the data has been loaded correctly or nto
print("Length of Training Data", len(X_train))
print("Number of Clases", num_classes)
print("Shape of Input", input_shape)


#Model Building Function

def build_dscnn_qat(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=64, kernel_size=(10, 4), strides=(2,2), padding="same", use_bias=False)(inputs)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    def dscnn_layer(x, deptwise_kernel=(3,3), pointwise_filters=64):
        x = keras.layers.DepthwiseConv2D(deptwise_kernel, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2D(pointwise_filters, kernel_size=(1,1), padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        return x
    
    x = dscnn_layer(x)
    x = dscnn_layer(x)
    x = dscnn_layer(x)
    x = dscnn_layer(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=inputs, outputs=output, name="dscnn_qat_model")
    return model


model = build_dscnn_qat(input_shape, num_classes)
model.summary()

initial_lr = 5e-4
opt = keras.optimizers.Adam(learning_rate=initial_lr)
model.compile(optimizer=opt,
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

# Model Quantisation
