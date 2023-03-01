from matplotlib import pyplot as plt


def show_acc():
    file = open('./log.txt', "r")
    s = file.read()

    acc = []
    val_acc = []

    idx = 0

    while True:
        idx = s.find(" accuracy: ", idx)
        if idx == -1:
            break

        nxt_space = s.find(" ", idx + 11)
        t = s[idx + 11 : nxt_space]

        num = float(t)
        acc.append(num)
        idx = nxt_space

    idx = 0

    while True:
        idx = s.find("val_accuracy: ", idx)
        if idx == -1:
            break

        nxt = s.find("\n", idx + 14)
        t = s[idx + 14 : nxt]

        num = float(t)
        val_acc.append(num)
        idx = nxt

    epochs_range = range(1, len(acc) + 1)

    plt.title('Accuracy')
    plt.xticks(epochs_range)
    plt.plot(epochs_range, acc, 'bo', label='Training acc')
    plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
    plt.legend()
    plt.show()




def show_loss():

    file = open('./log.txt', "r")
    s = file.read()

    loss = []
    val_loss = []

    idx = 0

    while True:
        z = ""
        idx = s.find(" loss: ", idx)
        if idx == -1:
            break
        z += str(idx)

        nxt_space = s.find(" ", idx + 7)
        t = s[idx + 7 : nxt_space]

        print("z = ", z, "t = ", t, "nxt = ", nxt_space)

        num = float(t)
        loss.append(num)
        idx = nxt_space

    idx = 0

    while True:
        z = ""
        idx = s.find(" val_loss: ", idx)
        if idx == -1:
            break
        z += str(idx)

        nxt_space = s.find(" ", idx + 11)
        t = s[idx + 11 : nxt_space]

        print("z = ", z, "t = ", t, "nxt = ", nxt_space)

        num = float(t)
        val_loss.append(num)
        idx = nxt_space

    epochs_range = range(1, len(loss) + 1)

    plt.title('Accuracy')
    plt.xticks(epochs_range)
    plt.plot(epochs_range, loss, 'bo', label='Training loss')
    plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
    plt.legend()
    plt.show()












