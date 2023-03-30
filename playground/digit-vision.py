#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist

from myml import (FlattenLayer, LinearLayer, ReluLayer, SigmoidLayer, SoftmaxLayer,
                  Sequence, CategoricalCrossEntropyLoss, Model)


def main():
    matplotlib.use('QtAgg')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Convert to float and normalize pixel values to 0..1 - this makes a huge difference
    x_train, x_test = x_train / 255.0, x_test / 255.0

    rng = default_rng()

    model = Model(
        rng,
        Sequence([
            FlattenLayer((28, 28)),
            LinearLayer((500, 28 * 28)),
            ReluLayer(),
            LinearLayer((100, 500)),
            ReluLayer(),
            LinearLayer((10, 100)),
            SoftmaxLayer(),
        ]),
        CategoricalCrossEntropyLoss(),
    )

    NROUNDS = 2998
    decile = (NROUNDS - 1) // 9
    n = x_train.shape[0]
    BATCH_SIZE = 100
    assert n % BATCH_SIZE == 0
    for i in range(NROUNDS):
        i0 = i * BATCH_SIZE % n
        i1 = i0 + BATCH_SIZE
        model.train(x_train[i0:i1], y_train[i0:i1])

        ## # print some facts about what the model is actually doing
        ## [[dzdx], [dzdy]] = layer.weights
        ## import math
        ## a = 180/math.pi * math.atan2(dzdy, dzdx)
        ## print("angle =", a) # there's no reason this shouldn't be 45
        ## r = -layer.b[0,0] / math.sqrt(dzdx ** 2 + dzdy ** 2)
        ## print("r =", r)  # and this should be 4

        ## print("accuracy (training data):", layer.accuracy(x_train, y_train))
        ## print("accuracy (test data):", layer.accuracy(x_test, y_test))

    n = x_test.shape[0]
    predicted_prob = model.apply(x_test)
    predictions = np.argmax(predicted_prob, axis=0)
    failures = np.arange(n)[predictions != y_test]
    print("{}/{} images misclassified ({:.1f}% accuracy)".format(
        len(failures), n, 100 * (n - len(failures)) / n))

    W, H = 16, 12

    fig, axs = plt.subplots(H, W, subplot_kw={'xticks': [], 'yticks': []})
    for cell, i in enumerate(failures):
        if cell >= H * W:
            break
        y = y_test[i]
        print("img #{} is {}, predicted {} with P={}, P[{}]={}".format(i, y, predictions[i], predicted_prob[predictions[i],i], y, predicted_prob[y,i]))
        axs.flat[cell].imshow(x_test[i], interpolation='bicubic')

    plt.show()


main()
