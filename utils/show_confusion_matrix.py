import tensorflow as tf
import numpy as np
from keras.utils import image_dataset_from_directory

image_width = 150
image_height = 150

train_path = './Dataset/train'
test_path = './Dataset/test'


def show_confusion_matrix_train(model):

    train_images = image_dataset_from_directory(
        train_path,
        image_size=(image_width, image_height),
        shuffle=False
    )

    labels = None
    first = True

    for images, _labels in train_images:
        if first:
            labels = _labels
            first = False
        else:
            labels = np.concatenate([labels, _labels])

    predictions = model.predict(train_images)
    predictions = predictions.argmax(axis=-1)

    print(tf.math.confusion_matrix(labels, predictions))


def show_confusion_matrix_test(model):

    test_images = image_dataset_from_directory(
        test_path,
        image_size=(image_width, image_height),
        shuffle=False
    )

    labels = None
    first = True

    for images, _labels in test_images:
        if first:
            labels = _labels
            first = False
        else:
            labels = np.concatenate([labels, _labels])

    predictions = model.predict(test_images)
    predictions = predictions.argmax(axis=-1)

    print(tf.math.confusion_matrix(labels, predictions))





















