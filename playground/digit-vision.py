#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist

from myml import (FlattenLayer, LinearLayer, ReluLayer, SigmoidLayer, SoftmaxLayer,
                  Sequence, CategoryCrossEntropyLoss, Model)


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
        CategoryCrossEntropyLoss(),
    )

    NROUNDS = 298
    decile = (NROUNDS - 1) // 9
    for i in range(NROUNDS):
        if i % decile == 0:
            print(".", end=None)
        model.train(x_train, y_train)

        ## # print some facts about what the model is actually doing
        ## [[dzdx], [dzdy]] = layer.weights
        ## import math
        ## a = 180/math.pi * math.atan2(dzdy, dzdx)
        ## print("angle =", a) # there's no reason this shouldn't be 45
        ## r = -layer.b[0,0] / math.sqrt(dzdx ** 2 + dzdy ** 2)
        ## print("r =", r)  # and this should be 4

        ## print("accuracy (training data):", layer.accuracy(x_train, y_train))
        ## print("accuracy (test data):", layer.accuracy(x_test, y_test))
    print(".")
    plt.show()


main()
