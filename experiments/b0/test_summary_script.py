import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# =========================================================
# Load preprocessed data
# =========================================================
DATA_PATH = "../../data/processed/gsc_mfcc40_ds_cnn.npz"  # <-- match preprocessing
data = np.load(DATA_PATH, allow_pickle=True)

X_train = data["X_train"]   # (N_train, 49, 40)
y_train = data["y_train"]   # (N_train,)
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]
label_to_index = data["label_to_index"].item()

# Add channel dimension for Conv2D: (time, freq, 1)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

num_classes = len(label_to_index)
input_shape = X_train.shape[1:]  # (49, 40, 1)

print("Input shape:", input_shape)
print("Num classes:", num_classes)
print("Train size:", X_train.shape[0])


# =========================================================
# DS-CNN model definition
# (one reasonable variant matching the KWS literature)
# =========================================================
def build_ds_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Conv front-end
    x = layers.Conv2D(
        filters=64,
        kernel_size=(10, 4),
        strides=(2, 2),
        padding="same",
        use_bias=False,
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise separable blocks
    def ds_block(x, depthwise_kernel=(3, 3), pointwise_filters=64):
        x = layers.DepthwiseConv2D(depthwise_kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(pointwise_filters, kernel_size=(1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # You can adjust number of blocks and filters to match specific DS-CNN variants
    x = ds_block(x, depthwise_kernel=(3, 3), pointwise_filters=64)
    x = ds_block(x, depthwise_kernel=(3, 3), pointwise_filters=64)
    x = ds_block(x, depthwise_kernel=(3, 3), pointwise_filters=64)
    x = ds_block(x, depthwise_kernel=(3, 3), pointwise_filters=64)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ds_cnn_kws")
    return model


model = build_ds_cnn(input_shape, num_classes)
model.summary()


initial_lr = 5e-4
opt = optimizers.Adam(learning_rate=initial_lr)

model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
# =========================================================
# Evaluation on test set
# =========================================================
model.load_weights("best_ds_cnn.weights.h5")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc * 100:.2f}%")
