import numpy as np
from scipy.misc import imread
import pandas as pd
from skimage import color, exposure, transform
import math
import os
import glob

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess(img):
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])  # Normalization on the V of the HSV img channel
    img = color.hsv2rgb(hsv)
    #
    # # Central square crop
    min_side = min(img.shape[:-1])
    center = img.shape[0] // 2, img.shape[1] // 2
    img = img[center[0] - min_side // 2:center[0] + min_side // 2,
              center[1] - min_side // 2:center[1] + min_side // 2,
              :]

    # Rescale
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # img = np.rollaxis(img, -1)

    return img


def get_class_from_path(img_path):
    return int(img_path.split('/')[-2])


def load_gtsrb():
    folder = "./dataset/GTSRB/Images/"
    test_folder_csv = "./dataset/GTSRB/Test/"
    test_csv = "GT-final_test.csv"
    imgs = []
    test_imgs = []
    labels = []
    test_labels = []
    all_imgs = glob.glob(os.path.join(folder, '*/*.ppm'))

    # Read csv from Test folder and load/preprocess all tests images
    csv = pd.read_csv(test_folder_csv + test_csv, sep=';')
    for index, row in csv.iterrows():
        test_labels.append(row['ClassId'])
        test_imgs.append(preprocess(imread(test_folder_csv + str(row['Filename']))))

    #  load/preprocess all training images
    np.random.shuffle(all_imgs)
    for img in all_imgs:
        imgs.append(preprocess(imread(img)))
        labels.append(get_class_from_path(img))

    # Creating Train numpy arrays
    X = np.array(imgs, dtype='float32')
    X = (X / 255.).astype(np.float32)
    Y = np.eye(NUM_CLASSES)[labels]  # One hot matrix

    # Creating Test numpy arrays
    XTest = np.array(test_imgs, dtype="float32")
    XTest = (XTest / 255.).astype(np.float32)
    YTest = test_labels

    return X, Y, XTest, YTest


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    :param X: input data, of shape (input size, number of examples)
    :param Y: true label vector
    :param mini_batch_size: size of the mini-batches, integer
    :param seed: the seed for randomization

    :return: mini_batches , a list of synchronous( mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # Number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1 : Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation].reshape((Y.shape[0], m))

    # Step 2 : Partition (shuffled_X, shuffled_Y) Minus the end case
    num_complete_minibatches = int(math.floor(m/mini_batch_size))  # Number of mini-batches of size mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches