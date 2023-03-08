from matplotlib import pyplot as plt


def show_image(image, title):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
