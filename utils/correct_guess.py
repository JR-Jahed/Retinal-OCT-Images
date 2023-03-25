import numpy as np


def correct_guess(predictions, test_labels):
    correct = 0
    incorrect = 0

    for i in range(len(test_labels)):

        label = test_labels[i]
        pred_label = np.argmax(predictions[i])

        if label == pred_label:
            correct += 1
        else:
            incorrect += 1

    print('correct: ', correct, 'incorrect: ', incorrect, 'test accuracy: ', (correct * 1.0) / len(predictions))