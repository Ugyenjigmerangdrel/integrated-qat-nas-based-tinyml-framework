import numpy as np

def load_data(DATA_PATH):
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

    

    return X_train, y_train, X_val, y_val, X_test, y_test, label_to_index, num_classes, input_shape