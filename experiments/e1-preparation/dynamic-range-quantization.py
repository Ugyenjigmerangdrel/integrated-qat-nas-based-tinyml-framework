import tensorflow as tf
from tensorflow_model_optimization.python.core.keras.compat import keras

model = keras.models.load_model("b2rs-best-dscnn-model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quant_model = converter.convert()

with open("b2rs-best-dscnn-model-int8.tflite", "wb") as f:
    f.write(quant_model)
