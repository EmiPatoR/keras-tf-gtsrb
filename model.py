import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import  AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from sklearn.cross_validation import train_test_split
from keras import regularizers
import keras.backend as K
import preprocess as p

K.set_image_dim_ordering('tf')  # Put the image format to (size, size, channel)
lr = 0.01  # Initial learning rate
_lambda = 0.01

def cnn_model2():
    model = Sequential()
    model.add(Conv2D(32, (7, 7), padding="same", input_shape=(p.IMG_SIZE, p.IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(p.NUM_CLASSES, activation='softmax'))
    return model

def cnn_model3():
    model = Sequential()

    model.add(Conv2D(32, (7, 7), padding="same", input_shape=(p.IMG_SIZE, p.IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (7, 7)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.35))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(p.NUM_CLASSES, activation='softmax'))
    return model


def cnn_model4():
    model = Sequential()

    model.add(Conv2D(32, (7, 7), padding="same", input_shape=(p.IMG_SIZE, p.IMG_SIZE, 3), kernel_regularizer=regularizers.l2(_lambda)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (7, 7), kernel_regularizer=regularizers.l2(_lambda)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), kernel_regularizer=regularizers.l2(_lambda)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (5, 5), kernel_regularizer=regularizers.l2(_lambda)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(_lambda)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))

    model.add(Conv2D(1024, (3, 3), kernel_regularizer=regularizers.l2(_lambda)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(Dropout(0.5))
    model.add(Dense(p.NUM_CLASSES, activation='softmax'))
    return model



# def lr_decay_rate(epoch):
#     return lr * (0.1 ** int(epoch/10))


def train_model(X, Y, Xval, Yval, batch_size, epochs, filename, cnn_model, datagen=None):
    model = cnn_model()
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])
    model.fit_generator(datagen.flow(X, Y, batch_size=batch_size),
              steps_per_epoch=X.shape[0],
              epochs=epochs,
              validation_data=(Xval, Yval),
              callbacks=[ModelCheckpoint(filename, save_best_only=True)])
    return model


def load_model(filename, cnn_model):
    model = cnn_model()
    model.load_weights("model2.h5")
    return model


def accuracy(model, XTest, YTest):
    # predict and evaluate
    y_pred = model.predict_classes(XTest)
    acc = np.sum(y_pred == YTest) / np.size(y_pred)
    print("Test accuracy = " + str(acc))


X, Y, XTest, YTest = p.load_gtsrb()

X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                  test_size=0.2, random_state=42)

datagen = ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10.)

datagen.fit(X_train)
model = train_model(X_train, Y_train, X_val, Y_val, 128, 30, "model3_regl2_data_augmentation.h5", cnn_model3, datagen)
accuracy(model, XTest, YTest)

