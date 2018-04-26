import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop, SGD

# saved_weights_name = 'CNN_Weights.h5'
saved_weights_name = 'SVM_Weights.h5'
images_shape = (48, 48, 1)


def get_model(model: str = 'CNN') -> Sequential:

    return {
        'CNN': get_cnn,
        'SVM': get_svm,
    }[model]()


def get_cnn() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # !- may kernel size cause crash if it did replace it with (3,3,1)
    # first conv layer
    model.add(Conv2D(32, 5, activation='relu', input_shape=images_shape))
    model.add(MaxPooling2D(2))

    # second conv layer
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(MaxPooling2D(2))

    # third conv layer
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(2))

    # fourth layer
    model.add(Conv2D(256, 2, activation='relu'))
    model.add(MaxPooling2D(2))

    # flatten the layers
    model.add(Flatten())

    # add fully connected layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def get_svm() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # TODO: Build SVM model

    # first conv layer
    model.add(Conv2D(32, 5, activation='relu', input_shape=images_shape))
    model.add(MaxPooling2D(2))

    # second conv layer
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(MaxPooling2D(2))

    # third conv layer
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(2))

    # fourth layer
    model.add(Conv2D(256, 2, activation='relu'))
    model.add(MaxPooling2D(2))

    # flatten the layers
    model.add(Flatten())

    # add fully connected layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, kernel_regularizer=keras.regularizers.l2(0.01)))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_hinge', metrics=["accuracy"])
    return model
