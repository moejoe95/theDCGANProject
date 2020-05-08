import glob
import numpy as np
from cv2 import cv2
import tensorflow as tf


def decode_img(img_file, resize):
    img = cv2.imread(img_file)
    if resize is not None:
        img = cv2.resize(img, resize)
    img = np.array(img) / 255.0
    img = (img[:, :, :3])
    img = np.float32(img)
    return img


def get_dataset_files(img_folders, resize=None):
    dataset_images = []

    for i, f in enumerate(img_folders):
        image_files = sorted(glob.glob(f + '/*'))
        dataset_images += image_files

    train_data = []
    for img_file in dataset_images:
        img = decode_img(img_file, resize)
        train_data.append(img)

    train_data = np.asarray(train_data)
    return train_data


def get_dataset_folders(img_folders, resize=None):
    return get_dataset_files(img_folders, resize)


def get_dataset_mnist():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    return train_images
