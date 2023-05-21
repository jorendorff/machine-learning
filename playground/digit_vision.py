#!/usr/bin/env python

import time
import numpy as np
from numpy.random import default_rng
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist

from myml import (FlattenLayer, LinearLayer, ReluLayer, SigmoidLayer, SoftmaxLayer,
                  Sequence, CategoricalCrossEntropyLoss, Model)
import mnist_mutate


def training_epochs(mnist_data, rng, num_epochs, batch_size):
    """Return iterable of epochs, which are iterables of (x, y) pairs."""
    (x_train, y_train) = mnist_data[0]
    n = len(x_train)

    def epoch(order):
        for j in range(0, n, batch_size):
            batch = order[j:j + batch_size]
            x = mnist_mutate.mutate_images(rng, x_train[batch])
            # Convert to float and normalize pixel values to 0..1 - this makes a huge difference
            x = x / 255.0
            yield x, y_train[batch]
    
    for i in range(num_epochs):
        yield epoch(rng.permutation(n))


def create_and_train_model(mnist_data):
    rng = default_rng()

    model = Model(
        rng,
        Sequence([
            FlattenLayer((28, 28)),
            LinearLayer((28 * 28, 500)),
            ReluLayer(),
            LinearLayer((500, 100)),
            ReluLayer(),
            LinearLayer((100, 10)),
            SoftmaxLayer(),
        ]),
        CategoricalCrossEntropyLoss(),
    )


    NUM_EPOCHS = 10
    t0 = time.time()
    model.train_epochs(training_epochs(mnist_data, rng, NUM_EPOCHS, batch_size=100))
    dt = time.time() - t0
    print(f"trained {NUM_EPOCHS} epochs in {dt:0.3} seconds")

    return model


def show_failures(mnist_data, model):
    (x_train, y_train), (x_test, y_test) = mnist_data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    n = x_test.shape[0]
    predicted_prob = model.apply(x_test)
    predictions = np.argmax(predicted_prob, axis=1)
    failures = np.arange(n)[predictions != y_test]

    print("{}/{} test images misclassified ({:.1f}% accuracy)".format(
        len(failures), n, 100 * (n - len(failures)) / n))

    # Sort failures from most severe to least
    predicted_prob_of_correct_answer = np.array([predicted_prob[i, y] for i, y in enumerate(y_test)])
    failures = [i for i in np.arange(n)[np.argsort(predicted_prob_of_correct_answer)]
                if repr(predictions[i]) != repr(y_test[i])]

    ## print("{}/{} test images misclassified ({:.1f}% accuracy)".format(
    ##     len(failures), n, 100 * (n - len(failures)) / n))

    matplotlib.use('QtAgg')
    W, H = 20, 15
    fig, axs = plt.subplots(H, W, subplot_kw={'xticks': [], 'yticks': []})
    for cell, i in enumerate(failures):
        if cell >= H * W:
            break
        y = y_test[i]
        print("img #{} is {}, predicted {} with P={}, P[{}]={}".format(i, y, predictions[i], predicted_prob[i,predictions[i]], y, predicted_prob[i,y]))
        axs.flat[cell].imshow(x_test[i], interpolation='bicubic')

    plt.show()


def show_a_batch(mnist_data):
    rng = default_rng()
    batch, y = next(next(training_epochs(mnist_data, rng, num_epochs=1, batch_size=100)))

    matplotlib.use('QtAgg')
    W, H = 10, 10
    fig, axs = plt.subplots(H, W, subplot_kw={'xticks': [], 'yticks': []})
    for cell, img in enumerate(batch):
        axs.flat[cell].imshow(img)
    plt.show()

def main():
    mnist_data = mnist.load_data()
    #show_a_batch(mnist_data)
    
    model = create_and_train_model(mnist_data)
    show_failures(mnist_data, model)



if __name__ == '__main__':
    main()
