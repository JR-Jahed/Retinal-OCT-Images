import os

from tensorflow import keras
from keras.applications import InceptionResNetV2
from keras.layers import Flatten
import numpy as np

from utils.get_last_file import get_last_file
from utils.load_model_data import load_model_data
from utils.model_checkpoint import MyModelCheckpoint
from utils.show_image import show_image

model_path = "./Saved Models/InceptionResNetV2"
batch_size = 32


class ModelInceptionResNetV2:
    def __init__(self, train_images, val_images, test_images, classes, input_shape):

        self.conv_base = InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

        self.model = keras.models.Sequential([
            self.conv_base,
            Flatten()
        ])

        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images

        self.classes = classes

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

        predictions = model.predict(self.test_images)

        self.correct_guess(predictions)

        while True:
            num = input("Enter a number between 0 and {}: ".format(self.test_images.samples - 1))
            num = int(num)

            if num == -1:
                break

            elif 0 <= num < self.test_images.samples:

                x = int(num / batch_size)
                y = num % batch_size

                show_image(self.test_images[x][0][y], "class: {}   prediction: {}".
                           format(self.classes[self.test_images[x][1][y]], self.classes[np.argmax(predictions[num])]))

            else:
                print("Please enter a correct number")

    def correct_guess(self, predictions):

        correct = 0
        incorrect = 0

        for i in range(len(self.test_images)):
            for j in range(len(self.test_images[i][1])):

                idx = i * batch_size + j

                label = self.test_images[i][1][j]
                pred_label = np.argmax(predictions[idx])

                if label == pred_label:
                    correct += 1
                else:
                    incorrect += 1

        print('correct: ', correct, 'incorrect: ', incorrect, 'test_accuracy: ', (correct * 1.0) / len(predictions))
