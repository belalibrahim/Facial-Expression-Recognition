import numpy as np
import pandas as pd
from keras.utils import np_utils
from matplotlib import pyplot as plt


def extract_images(image):
    return image.apply(lambda image: np.fromstring(image, sep=' '))


def get_image_values(image):
    return np.vstack(image.values).astype(np.float32).reshape(-1, 48, 48, 1)


def load_training_dataframe(location: str = "Datasets/train.csv"):
    """
    :param location:
    :return: tuple (x.y) where x is image array and y is the predictions
    """
    df = pd.read_csv(location)
    df['pixels'] = extract_images(df['pixels'])
    df = df.dropna()
    x_train = get_image_values(df['pixels'])
    y_train = np_utils.to_categorical(df['emotion'].values, 7)
    return x_train, y_train


def load_testing_dataframe(location: str = "Datasets/test.csv"):
    """
    :param location:
    :return:  x where x is image array
    """
    df = pd.read_csv(location)
    df['pixels'] = extract_images(df['pixels'])
    df = df.dropna()
    x_train = get_image_values(df['pixels'])
    return x_train


def view_image(image_arr, xs=None):
    arr = np.array(image_arr, dtype=np.uint8)
    arr.resize((48, 48))
    plt.imshow(arr, cmap='gray')
    for i in range(0, len(xs) - 1, 2):
        plt.scatter(xs[i], xs[i + 1], s=200, facecolors='none', edgecolors='r')
    plt.show()