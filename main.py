import os

import numpy as np
from tensorflow import keras
from keras.utils import image_dataset_from_directory

from models.model_inceptionv3 import ModelInceptionV3
from utils.get_last_file import get_last_file
from utils.logger import Logger

model_path = "./Saved Models/InceptionV3"
batch_size = 32

image_width = 150
image_height = 150

train_path = './Dataset/train'
val_path = './Dataset/val'
test_path = './Dataset/test'

# training = True
training = False


if __name__ == "__main__":

    image_size = (image_width, image_height)

    train_images = image_dataset_from_directory(
        train_path,
        image_size=image_size,
    )

    val_images = image_dataset_from_directory(
        val_path,
        image_size=image_size,
        shuffle=False
    )

    test_images = image_dataset_from_directory(
        test_path,
        image_size=image_size,
        shuffle=False
    )

    modelInceptionV3 = ModelInceptionV3(
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        input_shape=(image_width, image_height, 3)
    )

    modelInceptionV3.add_layer(keras.layers.Dense(256, activation='relu'))
    modelInceptionV3.add_layer(keras.layers.Dropout(.2))
    modelInceptionV3.add_layer(keras.layers.Dense(len(train_images.class_names), activation='softmax'))

    # image_batch, labels_batch = next(iter(train_images))
    # shape = np.expand_dims(image_batch[0], axis=0).shape
    # modelInceptionV3.model.build(input_shape=shape)
    # modelInceptionV3.conv_base.summary()
    # modelInceptionV3.model.summary()
    # exit(0)

    if training:
        os.makedirs(model_path, exist_ok=True)
        with Logger(os.path.join(model_path, 'log.txt')):
            modelInceptionV3.train()
    else:
        last_file = get_last_file(model_path)
        modelInceptionV3.test(os.path.join(last_file + '.h5'))













