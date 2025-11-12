import math
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time

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

def evaluate_saved_qat_model(
    weight_path,
    build_float_model_fn,  
    input_shape,
    num_classes,
    cfg,
    X_train, y_train, X_val, y_val, X_test, y_test,
    optimizer=None,
):

    float_model = build_float_model_fn(input_shape, num_classes, cfg)

    with tfmot.quantization.keras.quantize_scope():
        qat_model = tfmot.quantization.keras.quantize_model(float_model)

    if optimizer is None:
        optimizer = keras.optimizers.Adam(5e-4)

    qat_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    qat_model.load_weights(weight_path)

    train_loss, train_acc = qat_model.evaluate(X_train, y_train, verbose=0)
    val_loss,   val_acc   = qat_model.evaluate(X_val,   y_val,   verbose=0)
    test_loss,  test_acc  = qat_model.evaluate(X_test,  y_test,  verbose=0)

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc



def _quantize(sample_f32, scale, zero_point, dtype):
    """Quantize float32 -> uint8/int8 if needed."""
    if scale == 0:
        return sample_f32.astype(dtype)
    q = sample_f32 / scale + zero_point
    if dtype == np.uint8:
        q = np.clip(q, 0, 255).astype(np.uint8)
    elif dtype == np.int8:
        q = np.clip(np.round(q), -128, 127).astype(np.int8)
    else:
        # model expects float; just return float32
        q = sample_f32.astype(dtype)
    return q

def evaluate_int8_ptq_model(
    X_test, y_test, model_path, *,
    reps_per_sample=5,            # repeats per sample (after warmup)
    warmup_runs=3,                # warmup invokes (not timed)
    num_samples=200,              # cap #samples for timing to keep it quick
    num_threads=None              # e.g., 1 for deterministic, or 4 for speed
):
    model_size_kb = os.path.getsize(model_path) / 1024.0

    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_idx  = input_details[0]['index']
    out_idx = output_details[0]['index']

    in_scale,  in_zero  = input_details[0]['quantization']
    out_scale, out_zero = output_details[0]['quantization']
    in_dtype  = input_details[0]['dtype']
    out_dtype = output_details[0]['dtype']

    print(f"Input quantization:  scale={in_scale}  zero_point={in_zero}  dtype={in_dtype}")
    print(f"Output quantization: scale={out_scale} zero_point={out_zero} dtype={out_dtype}")

    # Evaluate accuracy (on the whole test set) while we’re here
    correct = 0
    for i in range(len(X_test)):
        x = _quantize(X_test[i:i+1].astype(np.float32), in_scale, in_zero, in_dtype)
        interpreter.set_tensor(in_idx, x)
        interpreter.invoke()
        logits = interpreter.get_tensor(out_idx)

        # dequantize if needed
        if out_scale != 0 and np.issubdtype(out_dtype, np.integer):
            logits = (logits.astype(np.float32) - out_zero) * out_scale

        if np.argmax(logits) == y_test[i]:
            correct += 1
    accuracy = correct / len(X_test)

    # -------- Inference latency benchmarking (TFLite interpreter) --------
    # Use a random subset to estimate latency robustly.
    n = min(num_samples, len(X_test))
    idxs = np.random.choice(len(X_test), size=n, replace=False)
    latencies_ms = []

    # Prepare one quantized tensor buffer to avoid reallocations in the loop
    sample0 = _quantize(X_test[idxs[0]:idxs[0]+1].astype(np.float32), in_scale, in_zero, in_dtype)
    interpreter.set_tensor(in_idx, sample0)
    for _ in range(warmup_runs):
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)

    for i in idxs:
        x = _quantize(X_test[i:i+1].astype(np.float32), in_scale, in_zero, in_dtype)

        # Repeat a few times per sample and average (reduces jitter)
        t0 = time.perf_counter()
        for _ in range(reps_per_sample):
            interpreter.set_tensor(in_idx, x)
            interpreter.invoke()
            _ = interpreter.get_tensor(out_idx)
        t1 = time.perf_counter()

        latencies_ms.append(((t1 - t0) / reps_per_sample) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    latency_mean  = float(latencies_ms.mean())
    latency_median= float(np.median(latencies_ms))
    latency_p95   = float(np.percentile(latencies_ms, 95))

    print(f"TFLite INT8 accuracy: {accuracy*100:.2f}%")
    print(f"Latency (ms/sample) — mean: {latency_mean:.3f}, median: {latency_median:.3f}, p95: {latency_p95:.3f}")
    print(f"Model size: {model_size_kb:.2f} KB")

    return {
        "accuracy": accuracy,
        "model_size_kb": model_size_kb,
        "latency_ms_mean": latency_mean,
        "latency_ms_median": latency_median,
        "latency_ms_p95": latency_p95,
        "n_timed_samples": int(n),
        "reps_per_sample": int(reps_per_sample),
        "threads": num_threads,
    }