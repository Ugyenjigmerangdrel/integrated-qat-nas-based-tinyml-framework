import math
import numpy as np
import tensorflow as tf


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
input_shape = X_train.shape[1:]  # (49, 40, 1)

print("Input shape:", input_shape)
print("Num classes:", num_classes)
print("Train size:", X_train.shape[0])

model = tf.keras.models.load_model("b2rs-best-dscnn-model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

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

with open("b2rs-best-dscnn-model-int8.tflite", "wb") as f:
    f.write(tflite_model)
