#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import matplotlib
import matplotlib.pyplot as plt

from myml import Sequence, SigmoidLayer, LinearLayer, LogisticLoss, Model


matplotlib.use('QtAgg')


# I don't know anything about the statistical properties of normal
# distributions, so we'll stick to two dimensions at first. When changing the
# number of dimensions, it might be worth thinking about whether the clusters
# are likely to overlap.
NDIMS = 2


def generate_data(rng, ntraining, ntesting):
    """Generate some data. You can ignore this. I'm sure this can be done less clumsily.

    Returns ((train_x, train_y), (test_x, test_y)) where both _x values are
    arrays of shape (n, 2); and both _y values are arrays of shape (n, 1)
    where each value is 0 or 1.
    """
    n = (ntraining + ntesting + 1) // 2
    a = rng.standard_normal(size=(NDIMS, n))
    b = 3.0 + rng.standard_normal(size=(NDIMS, n))

    a = np.r_[a, np.zeros((1, n))]
    b = np.r_[b, 1 + np.zeros((1, n))]

    points = np.c_[a, b]
    rng.shuffle(points, axis=1)

    def split_xy(data):
        return data[0:2], data[2:3]
    return split_xy(points[:,:ntraining]), split_xy(points[:,ntraining:ntraining + ntesting])


def main():
    rng = default_rng()
    (x_train, y_train), (x_test, y_test) = generate_data(rng, 800, 100)

    model = Model(
        rng,
        Sequence([
            LinearLayer((1, 2)),
            SigmoidLayer(),
        ]),
        LogisticLoss(),
    )

    fig, axs = plt.subplots(2, 5, subplot_kw={'projection': '3d'})

    viridis = matplotlib.colormaps['viridis']
    def graph(i):
        if i > 0:
            ax = axs.flat[i]
            h = np.array(history, dtype=float)
            for j in range(0, h.shape[0] - 1):
                hj = h[j:j+2]
                ax.plot(hj[:,0], hj[:,1], hj[:,2], color=viridis(h[j,3])) #color=h[:,3]

        ## plt.subplot(2, 5, i + 1)
        ## yh = model.apply(x_train)
        ## plt.scatter(x=x_train[0], y=x_train[1], c=yh)
        ## # superimpose the points we mispredict in red
        ## err_points = x_train[:, (yh[0] < 1/2) != (y_train[0] < 1/2)]
        ## plt.scatter(x=err_points[0], y=err_points[1], c='red', s=6)

    history = []

    NROUNDS = 100
    decile = (NROUNDS - 1) // 9
    for i in range(NROUNDS):
        if i % decile == 0:
            graph(i // decile)
        model.train(x_train, y_train)

        wx, wy, b = model.last_params
        loss = model.last_loss
        history.append([wx, wy, b, loss])

        ## # print some facts about what the model is actually doing
        ## [[dzdx], [dzdy]] = layer.weights
        ## import math
        ## a = 180/math.pi * math.atan2(dzdy, dzdx)
        ## print("angle =", a) # there's no reason this shouldn't be 45
        ## r = -layer.b[0,0] / math.sqrt(dzdx ** 2 + dzdy ** 2)
        ## print("r =", r)  # and this should be 4

        ## print("accuracy (training data):", layer.accuracy(x_train, y_train))
        ## print("accuracy (test data):", layer.accuracy(x_test, y_test))
    graph(9)
    plt.show()


main()
