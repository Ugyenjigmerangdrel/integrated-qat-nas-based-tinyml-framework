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
    print("⚠️ No GPU detected.")
    
# from tensorflow_model_optimization.python.core.keras.compat import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow_model_optimization as tfmot

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

def pooling_layer(x, selected_pooling):
    if selected_pooling=="gap":
        x = layers.GlobalAveragePooling2D()(x)
    elif selected_pooling=="max":
        x = layers.GlobalMaxPooling2D()(x)

    return x

def build_model(input_shape, num_classes, cfg):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=cfg["first_conv_filters"], kernel_size=cfg["first_conv_filters"], strides=cfg["first_conv_stride"], padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(cfg["num_dscnn_layers"]):
        x = build_dscnn_layer(x, depthwise_kernel=cfg["depthwise_kernel"], pointwise_filters=cfg["pointwise_filters"])

    if cfg["dropout_rate"] > 0:
        x = layers.Dropout(cfg["dropout_rate"])(x)

    x = pooling_layer(x, cfg["pooling_function"])
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="keyword-spotting-model") 

    return model

def generate_model_config(search_space):
    cfg = {}
    cfg["num_dscnn_layers"] = random.choice(search_space["num_dscnn_layers"])
    cfg["first_conv_filters"] = random.choice(search_space["first_conv_filters"])
    cfg["first_conv_kernel"] = random.choice(search_space["first_conv_kernel"])
    cfg["first_conv_stride"] = random.choice(search_space["first_conv_stride"])
    cfg["depthwise_kernel"] = random.choice(search_space["depthwise_kernel"])
    cfg["pointwise_filters"] = random.choice(search_space["pointwise_filters"])
    cfg["pooling_function"] = random.choice(search_space["pooling_function"])
    cfg["dropout_rate"] = random.choice(search_space["dropout_rate"])

    return cfg


# Search Strategy 1: Random Search 
def generate_unique_configs(search_space, n):
    seen = set()
    configs = []

    while len(configs) < n:
        print(f"Model Config {len(configs)} \n")
        cfg = generate_model_config(search_space)
        # serialize dict (sorted keys ensure deterministic string)
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
        print(cfg)
        print("---"*30)
    return configs

search_space = {
    "num_dscnn_layers": [2, 3, 4, 5, 6],

    "first_conv_filters": [32, 64],
    "first_conv_kernel": [(10, 4), (8, 4)],
    "first_conv_stride": [(1, 1), (2, 2)],

    "depthwise_kernel": [(3, 3), (5, 5)],
    "pointwise_filters": [32, 64, 96],

    "pooling_function": ["gap", "max"],  # "gap" = GlobalAveragePooling2D

    "dropout_rate": [0.0, 0.2, 0.3],
}

model_configs = generate_unique_configs(search_space, 20)

'''
How the search needs to work?
- single objective nas approach
- for each of the model configs generated through random config generator from search space
    - build the model
    - model summary
    - compile model
    - train model 10 epochs
    - evaluate model on accuracy
    - save the best model
    - check the inference latency
    - check the size of the model
    - build a summary results for comparison
    - clear the tensorflow backend
    - repeat
'''

count = 1
summary_results = {}

for cfg in model_configs:
    print(cfg)
    model = build_model(input_shape, num_classes, cfg)

    initial_lr = 5e-4
    opt = optimizers.Adam(learning_rate=initial_lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    batch_size = 100
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)
    total_iterations = 20000
    epochs = 10

    ckpt_callback = callbacks.ModelCheckpoint(
        "nas_dscnn.weights.h5",  
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    start_train = time.time()

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[ckpt_callback],
        validation_data=(X_val, y_val),
        verbose=1,
    )

    train_time = time.time() - start_train

    model.load_weights("nas_dscnn.weights.h5")

    # measure inference latency
    sample_input = X_test[:1]
    _ = model.predict(sample_input, verbose=0)

    runs = 50
    start_infer = time.time()
    for _ in range(runs):
        _ = model.predict(sample_input, verbose=0)
    end_infer = time.time()

    avg_latency = (end_infer - start_infer) / runs
    print(f"Average inference latency: {avg_latency*1000:.3f} ms per sample")

    # measure model size
    model.save("temp_model.h5", include_optimizer=False)
    model_size = os.path.getsize("temp_model.h5") / 1024  # KB
    print(f"Model size: {model_size:.2f} KB")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    summary_results[count] = [test_loss, test_acc, avg_latency, model_size]

    tf.keras.backend.clear_session()
    gc.collect()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    count+= 1


print(summary_results)

with open("random_nas_summary_results.json", "w") as f:
    json.dump(summary_results, f)






    








