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
from helpers.model import build_model, train_model, evaluate_int8_ptq_model

from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize


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


# Multi Objective Bayesian Optimization
search_space_skopt = [
    Integer(2, 6, name="num_dscnn_layers"),
    Categorical([32, 64], name="first_conv_filters"),
    Categorical(["(10,4)", "(8,4)"], name="first_conv_kernel"),
    Categorical(["(1,1)", "(2,2)"], name="first_conv_stride"),
    Categorical(["(3,3)", "(5,5)"], name="depthwise_kernel"),
    Categorical([32, 64, 96], name="pointwise_filters"),
    Categorical(["gap", "max"], name="pooling_function"),
    Categorical([0.0, 0.2, 0.3], name="dropout_rate"),
]

for dim in search_space_skopt:
    if hasattr(dim, 'categories'):
        for c in dim.categories:
            print(dim.name, c, type(c))

def parse_tuple_str(s):
    if isinstance(s, str) and s.startswith("(") and s.endswith(")"):
        return tuple(map(int, s.strip("()").split(",")))
    return s

summary_results = []

@use_named_args(search_space_skopt)
def objective(**params):
    for key in ["first_conv_kernel", "first_conv_stride", "depthwise_kernel"]:
        params[key] = parse_tuple_str(params[key])

    model = build_model(input_shape, num_classes, params)

    initial_lr = 5e-4
    opt = keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    ckpt_callback = keras.callbacks.ModelCheckpoint(
    "./models/mobo-best-dscnn-model.weights.h5",  
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    )


    batch_size = 100
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)
    total_iterations = 20000
    epochs = 10

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Planned epochs (approx 20K iterations): {epochs}")

    start_train = time.time()
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[ckpt_callback],
        verbose=1,
    )
    train_time = time.time() - start_train

    print(f"Traning Time: {train_time}")

    model.load_weights("./models/mobo-best-dscnn-model.weights.h5")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

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

    model.save("./models/mobo-best-dscnn-model.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for i in range(100):
            sample = X_train[i:i+1].astype("float32")
            yield [sample]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open("./models/mobo-best-dscnn-model.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"Pre PTQ Test accuracy: {test_acc * 100:.2f}%")
    int8_ptq_evals = evaluate_int8_ptq_model(
        X_test, y_test, "./models/mobo-best-dscnn-model.tflite", 
        reps_per_sample=5,            
        warmup_runs=3,                
        num_samples=200,              
        num_threads=None              
    )

    acc_fp32 = float(test_acc)
    lat_ms   = float(int8_ptq_evals["latency_ms_mean"])  
    size_kb  = float(model_size)
    acc_int8 = float(int8_ptq_evals["accuracy"])
    int8_drop = max(0.0, acc_fp32 - acc_int8)

    Lmax, Smax, Dmax = 10.0, 220.0, 0.02  
    lam1, lam2, lam3 = 20.0, 20.0, 30.0

    def relu(x): return x if x > 0 else 0
    J = (1.0 - acc_fp32) + lam1 * relu(lat_ms / Lmax - 1.0) + lam2 * relu(size_kb / Smax - 1.0) + lam3 * relu(int8_drop / Dmax - 1.0)

    tf.keras.backend.clear_session()
    gc.collect()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    summary_results.append([J, test_loss, acc_fp32, acc_int8, int8_drop, avg_latency, lat_ms, model_size, size_kb, train_time, params])

    with open("summary_results.txt", "a") as f:
        f.write(str([J, test_loss, acc_fp32, acc_int8, int8_drop, avg_latency, lat_ms, model_size, size_kb, train_time, params]) + "\n")

    return J  
    

result = gp_minimize(
    func=objective,
    dimensions=search_space_skopt,
    n_initial_points=5,
    n_calls=10
)


print("Best parameters:", result.x)
print("Best INT8 accuracy:", 1 - result.fun)

import pandas as pd
import json

# Convert to readable DataFrame
records = []
for params, func_val in zip(result.x_iters, result.func_vals):
    accuracy = 1 - func_val
    record = {
        "num_dscnn_layers": params[0],
        "first_conv_filters": params[1],
        "first_conv_kernel": params[2],
        "first_conv_stride": params[3],
        "depthwise_kernel": params[4],
        "pointwise_filters": params[5],
        "pooling_function": params[6],
        "dropout_rate": params[7],
        "int8_accuracy": accuracy
    }
    records.append(record)

df = pd.DataFrame(records)
print(df)

df.to_csv("./results/mobo_results.csv", index=False)
df.to_json("./results/mobo_results.json", orient="records", indent=2)








