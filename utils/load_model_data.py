import pickle
import keras


def load_model_data(mdl_path, opt_path):
    model = keras.models.load_model(mdl_path)

    with open(opt_path, 'rb') as fp:
        d = pickle.load(fp)
        epoch = d['epoch']
        opt = d['opt']
        return epoch, model, opt