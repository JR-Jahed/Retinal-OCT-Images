import pickle

import keras.models
from keras.callbacks import ModelCheckpoint


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        filepath = self._get_file_path(epoch, batch=32, logs=logs)
        model = keras.models.load_model(filepath)
        filepath = filepath.rsplit(".", 1)[0]
        filepath += ".pkl"

        with open(filepath, 'wb') as fp:
            pickle.dump(
                {
                    'opt': model.optimizer.get_config(),
                    'epoch': epoch + 1
                }, fp, protocol = pickle.HIGHEST_PROTOCOL
            )

        print('\nEpoch %02d: saving optimizer to %s' % (epoch + 1, filepath))
