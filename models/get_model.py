from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def get_model():
    conv_kerr_size = (3, 3)
    maxpool_kerr_size = (2, 2)
    model = Sequential()
    model.add(Conv2D(32, conv_kerr_size, activation='relu', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D(maxpool_kerr_size))
    model.add(BatchNormalization())

    model.add(Conv2D(64, conv_kerr_size, activation='relu'))
    model.add(MaxPooling2D(maxpool_kerr_size))
    model.add(BatchNormalization())

    model.add(Conv2D(128, conv_kerr_size, activation='relu'))
    model.add(MaxPooling2D(maxpool_kerr_size))
    model.add(BatchNormalization())

    model.add(Conv2D(256, conv_kerr_size, activation='relu'))
    # model.add(MaxPooling2D(maxpool_kerr_size))
    # model.add(BatchNormalization())

    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model
