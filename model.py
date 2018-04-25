import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop, SGD

saved_weights_name = 'Weights.h5'
images_shape = (48, 48, 1)


def get_model(model: str = 'CNN') -> Sequential:

    return {
        'CNN': get_cnn,
        'KNN': get_knn,
        'SVM': get_svm,
    }[model]()


def get_cnn() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # !- may kernel size cause crash if it did replace it with (3,3,1)
    # first conv layer
    model.add(Conv2D(32, 3, activation='sigmoid', input_shape=images_shape))
    model.add(MaxPooling2D(2))

    # second conv layer
    model.add(Conv2D(64, 3, activation='sigmoid'))
    model.add(MaxPooling2D(2))

    # third conv layer
    model.add(Conv2D(128, 3, activation='sigmoid'))
    model.add(MaxPooling2D(2))

    # fourth layer
    model.add(Conv2D(256, 2, activation='sigmoid'))
    model.add(MaxPooling2D(2))

    # flatten the layers
    model.add(Flatten())

    # add fully connected layers
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(7, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def get_knn() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # TODO: Build KNN model

    return model


def get_svm() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # TODO: Build SVM model

    return model
