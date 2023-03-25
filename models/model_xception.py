import os

from tensorflow import keras
from keras.applications import Xception
from keras.layers import Flatten
import numpy as np

from utils.correct_guess import correct_guess
from utils.get_last_file import get_last_file
from utils.load_model_data import load_model_data
from utils.model_checkpoint import MyModelCheckpoint
from utils.show_image import show_image

model_path = "./Saved Models/Xception"
batch_size = 32


class ModelXception:
    def __init__(self, train_images, val_images, test_images, input_shape):

        self.conv_base = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

        self.conv_base.trainable = False

        resize_rescale = keras.models.Sequential([
            keras.layers.Resizing(width=input_shape[0], height=input_shape[1]),
            keras.layers.Rescaling(1. / 255)
        ])

        self.model = keras.models.Sequential([
            resize_rescale,
            self.conv_base,
            Flatten()
        ])

        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images

        self.classes = train_images.class_names

    def add_layer(self, layer):
        self.model.add(layer)

    def train(self):

        initial_epoch = 0

        last_file = get_last_file(model_path)

        if last_file == "log":

            """ This is the first epoch """

            model = self.model
            model.compile(
                optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )

        else:

            mdl_path = os.path.join(model_path, last_file + '.h5')
            opt_path = os.path.join(model_path, last_file + '.pkl')
            initial_epoch, model, opt = load_model_data(mdl_path, opt_path)

            """Uncomment the following lines after 5 epochs"""

            # trainable = False
            #
            # for layer in model.layers:
            #     if layer.name == "xception":
            #         for _layer in layer.layers:
            #             if _layer.name == "block13_sepconv2":
            #                 trainable = True
            #
            #             _layer.trainable = trainable

            model.compile(
                optimizer=keras.optimizers.Adam.from_config(opt),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )

        model_ckpt = MyModelCheckpoint(
            os.path.join(model_path, 'model{epoch:02d}.h5'),
            monitor="val_accuracy",
            mode="max",
        )

        model.fit(
            self.train_images,
            validation_data=self.val_images,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + 1,
            callbacks=[model_ckpt]
        )

    def test(self, model_name):

        mdl_path = os.path.join(model_path, model_name)

        model = keras.models.load_model(mdl_path)

        test_images = None
        test_labels = None

        first = True

        for images, labels in self.test_images:
            if first:
                test_images = images
                test_labels = labels
                first = False
            else:
                test_images = np.concatenate([test_images, images])
                test_labels = np.concatenate([test_labels, labels])

        predictions = model.predict(test_images)

        correct_guess(predictions, test_labels)

        while True:
            num = input("Enter a number between 0 and {}: ".format(len(test_images) - 1))
            num = int(num)

            if num == -1:
                break

            elif 0 <= num < len(test_images):
                show_image(test_images[num] / 255, "class: {}   prediction: {}".
                           format(self.classes[test_labels[num]], self.classes[np.argmax(predictions[num])]))

            else:
                print("Please enter a correct number")


