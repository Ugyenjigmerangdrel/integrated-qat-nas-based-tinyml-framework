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

from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize


SEED = 42  
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "../../../data/processed/gsc_mfcc40_ds_cnn.npz"

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

'''
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
'''


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

# exit()


def evaluate_int8_ptq_model(X_test, y_test, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
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

    return accuracy

def parse_tuple_str(s):
    if isinstance(s, str) and s.startswith("(") and s.endswith(")"):
        return tuple(map(int, s.strip("()").split(",")))
    return s


summary_results = []

@use_named_args(search_space_skopt)
def objective(**params):
    # Build and train your model using params
    for key in ["first_conv_kernel", "first_conv_stride", "depthwise_kernel"]:
        params[key] = parse_tuple_str(params[key])

    model = build_model(input_shape, num_classes, params)
    

    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer=opt,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    q_aware_model.summary()

    # Model Quantisation

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
        "qat_sobo_dscnn_best_model.weights.h5",  
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
    start_train = time.time()
    history = q_aware_model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[lr_callback, ckpt_callback],
        verbose=1,
    )
    train_time = time.time() - start_train

    print(f"Traning Time: {train_time}")

    val_acc = history.history["val_accuracy"][-1]
    

    q_aware_model.load_weights("qat_sobo_dscnn_best_model.weights.h5")
    test_loss, test_acc = q_aware_model.evaluate(X_test, y_test, verbose=0)
    
    # measure inference latency
    sample_input = X_test[:1]
    _ = q_aware_model.predict(sample_input, verbose=0)

    runs = 50
    start_infer = time.time()
    for _ in range(runs):
        _ = q_aware_model.predict(sample_input, verbose=0)
    end_infer = time.time()

    avg_latency = (end_infer - start_infer) / runs
    print(f"Average inference latency: {avg_latency*1000:.3f} ms per sample")

    # measure model size
    q_aware_model.save("temp_model.h5", include_optimizer=False)
    model_size = os.path.getsize("temp_model.h5") / 1024  # KB
    print(f"Model size: {model_size:.2f} KB")

    q_aware_model.save("qat-sobo-model-dscnn.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for i in range(100):
            sample = X_train[i:i+1].astype("float32")
            yield [sample]

    converter.representative_dataset = representative_data_gen

    # Uncomment one depending on your desired precision
    # converter.target_spec.supported_types = [tf.float16]  # FP16 quantization

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open("qat-sobo-model-dscnn-int8.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"Pre PTQ Test accuracy: {test_acc * 100:.2f}%")
    int8_accuracy = evaluate_int8_ptq_model(X_test, y_test, "qat-sobo-model-dscnn-int8.tflite")
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    summary_results.append([test_loss, test_acc, avg_latency, model_size, train_time, int8_accuracy, params])

    with open("summary_results.txt", "a") as f:
        f.write(str([test_loss, test_acc, avg_latency, model_size, train_time, int8_accuracy, params]) + "\n")


    return 1 - int8_accuracy  

cfg = {"num_dscnn_layers": 2, "first_conv_filters": 64, "first_conv_kernel": (10, 4), "first_conv_stride": (2, 2), "depthwise_kernel": (5, 5), "pointwise_filters": 96, "pooling_function": "max", "dropout_rate": 0.0}

x0 = [
    [2, 64, (10,4), (2,2), (5,5), 96, "max", 0.0]]


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

df.to_csv("bayesian_optimization_results.csv", index=False)
df.to_json("bayesian_optimization_results.json", orient="records", indent=2)
