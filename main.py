import os

import numpy as np
from tensorflow import keras
from keras.utils import image_dataset_from_directory

from models.model_inceptionresnetv2 import ModelInceptionResNetV2
from utils.get_last_file import get_last_file
from utils.logger import Logger

model_path = "./Saved Models/Xception"
batch_size = 32

image_width = 150
image_height = 150

train_path = './Dataset/train'
val_path = './Dataset/val'
test_path = './Dataset/test'


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

    modelInceptionResNetV2 = ModelInceptionResNetV2(
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        input_shape=(image_width, image_height, 3)
    )

    modelInceptionResNetV2.add_layer(keras.layers.Dense(256, activation='relu'))
    modelInceptionResNetV2.add_layer(keras.layers.Dropout(.2))
    modelInceptionResNetV2.add_layer(keras.layers.Dense(len(train_images.class_names), activation='softmax'))

    # image_batch, labels_batch = next(iter(train_images))
    # shape = np.expand_dims(image_batch[0], axis=0).shape
    # modelInceptionResNetV2.model.build(input_shape=shape)
    # modelInceptionResNetV2.conv_base.summary()
    # modelInceptionResNetV2.model.summary()
    # exit(0)

    # training = True
    training = False

    if training:
        os.makedirs(model_path, exist_ok=True)
        with Logger(os.path.join(model_path, 'log.txt')):
            modelInceptionResNetV2.train()

        last_file = get_last_file(model_path)
        modelInceptionResNetV2.test(os.path.join(last_file + '.h5'))
    else:
        last_file = get_last_file(model_path)
        modelInceptionResNetV2.test(os.path.join(last_file + '.h5'))
