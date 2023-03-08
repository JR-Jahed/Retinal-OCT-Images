import os

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

from models.model_inceptionresnetv2 import ModelInceptionResNetV2
from utils.get_last_file import get_last_file
from utils.logger import Logger

model_path = "./Saved Models/InceptionResNetV2"
batch_size = 32

image_width = 150
image_height = 150

train_path = './Dataset/train'
val_path = './Dataset/val'
test_path = './Dataset/test'

training = True
# training = False


if __name__ == "__main__":

    image_size = (image_width, image_height)
    image_gen = ImageDataGenerator(
        rescale=1 / 255.,
    )

    train_images = image_gen.flow_from_directory(
        train_path,
        target_size=image_size,
        class_mode='sparse',
    )

    val_images = image_gen.flow_from_directory(
        val_path,
        target_size=image_size,
        class_mode='sparse',
        shuffle=False
    )

    test_images = image_gen.flow_from_directory(
        test_path,
        target_size=image_size,
        class_mode='sparse',
        shuffle=False
    )

    classes = {v: k for k, v in train_images.class_indices.items()}

    modelInceptionResNetV2 = ModelInceptionResNetV2(
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        classes=classes,
        input_shape=(image_width, image_height, 3)
    )

    modelInceptionResNetV2.add_layer(keras.layers.Dense(256, activation='relu'))
    modelInceptionResNetV2.add_layer(keras.layers.Dropout(.2))
    modelInceptionResNetV2.add_layer(keras.layers.Dense(len(classes), activation='softmax'))

    trainable = False

    for layer in modelInceptionResNetV2.conv_base.layers:
        if layer.name == "block8_10_conv":
            trainable = True

        layer.trainable = trainable

    if training:
        os.makedirs(model_path, exist_ok=True)
        with Logger(os.path.join(model_path, 'log.txt')):
            modelInceptionResNetV2.train()
    else:
        last_file = get_last_file(model_path)
        modelInceptionResNetV2.test(os.path.join(last_file + '.h5'))













