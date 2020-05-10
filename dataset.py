import glob
import numpy as np
from cv2 import cv2
import tensorflow as tf
from pathlib import Path


def decode_img(img_file, resize):
    img = cv2.imread(img_file)
    if resize is not None:
        img = cv2.resize(img, resize)
    img = np.array(img) / 255.0
    img = (img[:, :, :3])
    img = np.float32(img)
    return img


def get_dataset_files(img_folders, resize=None, recursive=False):
    dataset_images = []

    for i, f in enumerate(img_folders):
        if recursive:
            for path in Path(f).rglob('*.jpg'):
                dataset_images.append(str(path))
        else:
            image_files = sorted(glob.glob(f + '/*'))
            dataset_images += image_files

    dataset_images = dataset_images[:5000]
    train_data = np.empty((5000, 128, 128, 3), dtype=np.float32)

    for i, img_file in enumerate(dataset_images):
        if i % 10000 == 0:
            print('read in ' + str(i))
        img = decode_img(img_file, resize)
        train_data[i] = img

    return train_data


def get_dataset_folders(img_folders, resize=None, recursive=False):
    return get_dataset_files(img_folders, resize, recursive)


def get_dataset_mnist():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    return train_images
