#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('QtAgg')


# I don't know anything about the statistical properties of normal
# distributions, so we'll stick to two dimensions at first. When changing the
# number of dimensions, it might be worth thinking about whether the clusters
# are likely to overlap.
NDIMS = 2


def generate_data(rng, ntraining, ntesting):
    """Generate some data. You can ignore this. I'm sure this can be done less clumsily.

    Returns ((train_x, train_y), (test_x, test_y)) where both _x values are
    arrays with n rows and 2 columns; and both _y values are arrays with n rows
    and 1 column that's either 0 or 1.
    """
    n = (ntraining + ntesting + 1) // 2
    a = rng.standard_normal(size=(n, NDIMS))
    b = 8.0 + rng.standard_normal(size=(n, NDIMS))

    a = np.c_[a, np.zeros((n, 1))]
    b = np.c_[b, 1 + np.zeros((n, 1))]

    points = np.r_[a, b]
    rng.shuffle(points)

    def split_xy(data):
        return data[:,:-1], data[:,-1:]
    return split_xy(points[:ntraining]), split_xy(points[ntraining:ntraining + ntesting])


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


class DenseLayer:
    """ A dense layer of perceptrons, like tensorflow.keras.layers.Dense. """

    def __init__(self, rng, shape):
        num_inputs, num_outputs = shape
        # I can't believe backpropagation will work without randomized initial
        # parameters. I'm missing something but so be it, for now.
        self.weights = 0.2 + 0.1 * rng.standard_normal(shape)
        self.b = 0.2 + 0.1 * rng.standard_normal((1, num_outputs))

    def __call__(self, data):
        return sigmoid(data @ self.weights + self.b)

    def accuracy(self, x, y):
        """Percentage of the examples in x where we correctly predict y.

        Each point is counted as either right or wrong.
        """
        yh = self(x)
        return np.mean((yh < 1/2) == (y < 1/2))

    def loss(self, x, y):
        """ The loss function is what training seeks to minimize. """
        # Not to say it actually works.
        yh = self(x)
        err = yh ** y * (1 - yh) ** (1 - y)
        return np.mean(-np.log(err))

    def train(self, x, y, learn_rate):
        db, dw = self.derivatives(x, y)
        self.b -= learn_rate * db
        self.weights -= learn_rate * dw

    def derivatives(self, x, y):
        """Return the partial derivatives ∂L/∂b and ∂L/∂w.

        That is, return a characterazation of how tweaking each parameter of
        the model would affect the loss.

        Returns two arrays: the first, ∂L/∂b, is an array the same size as
        self.b, and each element tells the partial derivative of the loss with
        respect to the corresponding element of b.
        """

        # TODO: this code does not follow the conventions in the video course:
        # the linear combination should be called z (not m), and should be a
        # column vector, not a row vector. Also the weights are called w.
        ni, no = self.weights.shape
        n = x.shape[0] # number of training samples
        assert x.shape == (n, ni)
        assert y.shape == (n, no)

        m = x @ self.weights + self.b
        assert m.shape == (n, no)
        yh = sigmoid(m)
        if not np.all(np.isfinite(yh)):
            raise ValueError("non-finite output")
        assert yh.shape == (n, no)
        err = yh ** y * (1 - yh) ** (1 - y)
        loss = np.mean(-np.log(err))  # scalar
        ## print("loss:", loss)
        if not np.isfinite(loss):
            raise ValueError("non-finite loss")

        # Following the convention, for ∂L/∂x we write `dx`.
        derr = 1 / len(x) * -1 / err
        assert derr.shape == (n, no)
        dyh = derr * (2 * y - 1)
        assert dyh.shape == (n, no)
        dm = dyh * -1 / (1 + np.exp(-m))**2 * -np.exp(-m)
        assert dm.shape == (n, no)
        db = np.sum(dm, axis=0, keepdims=True)
        assert db.shape == (1, no)
        dw = x.T @ dyh
        assert dw.shape == (ni, no)

        return db, dw


def main():
    rng = default_rng()
    (x_train, y_train), (x_test, y_test) = generate_data(rng, 800, 100)
    layer = DenseLayer(rng, (2, 1))

    def graph(i, out):
        plt.subplot(2, 5, i + 1)
        plt.scatter(
            x=x_train[:,0],
            y=x_train[:,1],
            c=out, # color is the output of the network
        )
        # superimpose the points we mispredict in red
        err_points = x_train[(out[:,0] < 1/2) != (y_train[:,0] < 1/2),:]
        plt.scatter(
            x=err_points[:,0],
            y=err_points[:,1],
            c='red',
            s=8,
        )

    plt.figure()
    MOD = 1000
    N = 9 * MOD
    for i in range(N):
        if i % MOD == 0:
            graph(i // MOD, layer(x_train))
        learn_rate = 0.05
        layer.train(x_train, y_train, learn_rate)

        ## # print some facts about what the model is actually doing
        ## [[dzdx], [dzdy]] = layer.weights
        ## import math
        ## a = 180/math.pi * math.atan2(dzdy, dzdx)
        ## print("angle =", a) # there's no reason this shouldn't be 45
        ## r = -layer.b[0,0] / math.sqrt(dzdx ** 2 + dzdy ** 2)
        ## print("r =", r)  # and this should be 4

        ## print("accuracy (training data):", layer.accuracy(x_train, y_train))
        ## print("accuracy (test data):", layer.accuracy(x_test, y_test))
        if layer.loss(x_test, y_test) < 0.02:
            break  # eh, good enough
    graph(9, layer(x_train))
    plt.show()


main()
