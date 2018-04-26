import numpy as np
import pandas as pd
from keras.utils import to_categorical
from matplotlib import pyplot as plt


def extract_images(image):
    return image.apply(lambda image: np.fromstring(image, sep=' '))


def get_image_values(image,model):
    if(model):
        return np.vstack(image.values).astype(np.float32).reshape(-1, 48*48)
    else:
        return np.vstack(image.values).astype(np.float32).reshape(-1, 48,48,1)


def load_training_dataframe(location: str = "Datasets/train.csv",model = 0):
    """
    :param location:
    :return: tuple (x.y) where x is image array and y is the predictions
    """
    df = pd.read_csv(location)
    df['pixels'] = extract_images(df['pixels'])
    df = df.dropna()
    x_train = get_image_values(df['pixels'],model)
    y_train = to_categorical(df['emotion'].values, 2)
    return x_train, y_train


def load_testing_dataframe(location: str = "Datasets/test.csv",model = 0):
    """
    :param location:
    :return:  x where x is image array
    """
    df = pd.read_csv(location)
    df['pixels'] = extract_images(df['pixels'])
    df = df.dropna()
    x_train = get_image_values(df['pixels'],model)
    y_train = to_categorical(df['emotion'].values, 2)
    return x_train, y_train


def view_image(image_arr, actual=None, prediction=None):
    arr = np.array(image_arr, dtype=np.uint8)
    arr.resize((48, 48))
    plt.imshow(arr, cmap='gray')
    if prediction == 0:
        pred = 'Happy'
    else:
        pred = "Sad"

    if actual[1] == 0:
        act = 'Happy'
    else:
        act = "Sad"
    plt.title('Actual: ' + str(act) + '  Prediction: ' + str(pred))
    plt.show()
