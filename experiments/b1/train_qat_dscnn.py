import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks


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
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=64, kernel_size=(10, 4), strides=(2,2), padding="same", use_bias=False)(input)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    def dscnn_layer(x, deptwise_kernel=(3,3), pointwise_filters=64):
        x = layers.DepthwiseConv2D(deptwise_kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(pointwise_filters, kernel_size=(1,1), padding="same", use_bias=False)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return x
    
    x = dscnn_layer(x)
    x = dscnn_layer(x)
    x = dscnn_layer(x)
    x = dscnn_layer(x)

    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=input, outputs=output, name="dscnn_qat_model")
    return model


model = build_dscnn_qat(input_shape, num_classes)
model.summary()