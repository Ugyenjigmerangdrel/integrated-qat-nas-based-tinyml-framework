import math
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
import pandas as pd

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

def train_model(model, optimizer, X_train, y_train, X_val, y_val, X_test, y_test, checkpoint_path, model_path):

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    def lr_schedule(epoch):
        # approximate: if epoch corresponds to >= 10K iterations, drop LR
        # Steps per epoch = N_train / batch_size
        # 10K iterations = 10000 * batch_size samples
        # epoch_switch = (10000 * batch_size) / N_train
        batch_size = 100
        steps_per_epoch = max(1, X_train.shape[0] // batch_size)
        epoch_switch = math.floor(10000 / steps_per_epoch)

        if epoch >= epoch_switch:
            return 1e-4
        else:
            return 5e-4


    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    ckpt_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,  
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    batch_size = 100
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)
    total_iterations = 20000
    epochs = math.ceil(total_iterations / steps_per_epoch)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Planned epochs (approx 20K iterations): {epochs}")

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[lr_callback, ckpt_callback],
        verbose=1,
    )

    model.load_weights(checkpoint_path)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    model.save(model_path)

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

def train_model(model, optimizer, X_train, y_train, X_val, y_val, X_test, y_test, checkpoint_path, model_path):

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    def lr_schedule(epoch):
        # approximate: if epoch corresponds to >= 10K iterations, drop LR
        # Steps per epoch = N_train / batch_size
        # 10K iterations = 10000 * batch_size samples
        # epoch_switch = (10000 * batch_size) / N_train
        batch_size = 100
        steps_per_epoch = max(1, X_train.shape[0] // batch_size)
        epoch_switch = math.floor(10000 / steps_per_epoch)

        if epoch >= epoch_switch:
            return 1e-4
        else:
            return 5e-4


    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    ckpt_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,  
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    batch_size = 100
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)
    total_iterations = 20000
    epochs = math.ceil(total_iterations / steps_per_epoch)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Planned epochs (approx 20K iterations): {epochs}")

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[lr_callback, ckpt_callback],
        verbose=1,
    )

    model.load_weights(checkpoint_path)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    model.save(model_path)

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

def train_qat_model(model, optimizer, X_train, y_train, X_val, y_val, X_test, y_test, checkpoint_path, model_path):

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)

    q_aware_model.compile(optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    def lr_schedule(epoch):
        # approximate: if epoch corresponds to >= 10K iterations, drop LR
        # Steps per epoch = N_train / batch_size
        # 10K iterations = 10000 * batch_size samples
        # epoch_switch = (10000 * batch_size) / N_train
        batch_size = 100
        steps_per_epoch = max(1, X_train.shape[0] // batch_size)
        epoch_switch = math.floor(10000 / steps_per_epoch)

        if epoch >= epoch_switch:
            return 1e-4
        else:
            return 5e-4


    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    ckpt_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,  
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    batch_size = 100
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)
    total_iterations = 20000
    epochs = math.ceil(total_iterations / steps_per_epoch)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Planned epochs (approx 20K iterations): {epochs}")

    history = q_aware_model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[lr_callback, ckpt_callback],
        verbose=1,
    )

    q_aware_model.load_weights(checkpoint_path)
    train_loss, train_acc = q_aware_model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = q_aware_model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = q_aware_model.evaluate(X_test, y_test, verbose=0)

    q_aware_model.save(model_path)

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


def evaluate_saved_model(model_path, X_train, y_train, X_val, y_val, X_test, y_test):

    model = keras.models.load_model(model_path)
    # model.summary()

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

def evaluate_saved_qat_model(model_path, X_train, y_train, X_val, y_val, X_test, y_test):
    with tfmot.quantization.keras.quantize_scope():
        model = keras.models.load_model(model_path)
    # model.summary()

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
