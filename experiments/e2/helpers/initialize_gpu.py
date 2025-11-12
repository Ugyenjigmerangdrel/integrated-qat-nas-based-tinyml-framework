import tensorflow as tf

def initialize_gpu():
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return f"Enabled memory growth for {len(gpus)} GPU(s)."
        except RuntimeError as e:
            return e
    else:
        return "No GPU detected."