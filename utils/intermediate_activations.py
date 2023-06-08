from tensorflow import keras
from keras import utils
import numpy as np
import matplotlib.pyplot as plt


def show_intermediate_activations(model_path, image_path, show_all: bool):

    model = keras.models.load_model(model_path)
    model.summary()

    for layer in model.layers:
        if layer.name == "xception":
            model = layer

    img = utils.load_img(image_path, target_size=(150, 150))
    img_tensor = utils.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0

    layer_outputs = [layer.output for layer in model.layers[1:]]

    layer_index = []

    idx = 0
    cnt = 0

    for layer in layer_outputs:
        if layer.name.find("conv") != -1:
            cnt += 1
            layer_index.append(idx)
        idx += 1

    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    if show_all:
        layer_names = []
        for layer in model.layers[:8]:
            layer_names.append(layer.name)

        images_per_row = 16

        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]

            size = layer_activation.shape[1]

            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))

            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()
    else:
        first_layer_activation = activations[0]

        plt.matshow(activations[layer_index[0]][0, :, :, 0], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[0]][0, :, :, 3], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[0]][0, :, :, 6], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[0]][0, :, :, 10], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[1]][0, :, :, 5], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[1]][0, :, :, 10], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[1]][0, :, :, 15], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[1]][0, :, :, 20], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[2]][0, :, :, 0], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[2]][0, :, :, 1], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[2]][0, :, :, 11], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[3]][0, :, :, 0], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[4]][0, :, :, 0], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[5]][0, :, :, 0], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[6]][0, :, :, 0], cmap='viridis')
        plt.axis("off")
        plt.matshow(activations[layer_index[7]][0, :, :, 0], cmap='viridis')
        plt.axis("off")

        plt.matshow(first_layer_activation[0, :, :, 27], cmap='viridis')
        plt.matshow(first_layer_activation[0, :, :, 28], cmap='viridis')
        plt.matshow(first_layer_activation[0, :, :, 29], cmap='viridis')
        plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
        plt.matshow(first_layer_activation[0, :, :, 31], cmap='viridis')
        plt.show()
