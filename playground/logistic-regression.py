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


def split_xy(data):
    return data[:,:-1], data[:,-1:]


def generate_data(rng, ntraining, ntesting):
    # We'll make two clusters.
    n = (ntraining + ntesting + 1) // 2
    a = rng.standard_normal(size=(n, NDIMS))
    b = 8.0 + rng.standard_normal(size=(n, NDIMS))

    a = np.c_[a, np.zeros((n, 1))]
    b = np.c_[b, 1 + np.zeros((n, 1))]

    points = np.r_[a, b]
    rng.shuffle(points)
    return split_xy(points[:ntraining]), split_xy(points[ntraining:ntraining + ntesting])


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


class DenseLayer:
    """ A dense layer of perceptrons. """
    def __init__(self, rng, shape):
        num_inputs, num_outputs = shape
        # I can't believe backpropagation will work without randomized initial
        # parameters. I'm missing something but so be it, for now.
        self.weights = 0.2 + 0.1 * rng.standard_normal(shape)
        self.b = 0.2 + 0.1 * rng.standard_normal((1, num_outputs))

    def __call__(self, data):
        return sigmoid(data @ self.weights + self.b)

    def accuracy(self, x, y):
        yh = self(x)
        return np.mean((yh < 1/2) == (y < 1/2))

    def loss(self, yh, y):
        return np.mean(-np.log(yh ** y * (1 - yh) ** (1 - y)))

    def train(self, x, y, learn_rate):
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
        assert yh.shape == (n, no)
        err = yh ** y * (1 - yh) ** (1 - y)
        loss = np.mean(-np.log(err))  # scalar
        print("loss:", loss)

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

        self.b -= learn_rate * db
        self.weights -= learn_rate * dw


rng = default_rng()
(x_train, y_train), (x_test, y_test) = generate_data(rng, 8000, 1000)
layer = DenseLayer(rng, (2, 1))

def graph(i, out):
    plt.subplot(2, 5, i + 1)
    plt.scatter(
        x=x_train[:,0],
        y=x_train[:,1],
        c=out,
        #['red' if y_train[i,0] < 1/2 else 'blue'
        #   for i in range(len(y_train))]
    )
    bad_points = x_train[(out[:,0] < 1/2) != (y_train[:,0] < 1/2),:]
    plt.scatter(
        x=bad_points[:,0],
        y=bad_points[:,1],
        c='red',
        s=8,
    )

plt.figure()
N = 4500
for i in range(N):
    if i % 500 == 0:
        graph(i // 500, layer(x_train))
    layer.train(x_train, y_train, 0.03 + i / N * 0.15)
    acc = layer.accuracy(x_test, y_test)
    print("accuracy:", acc)
    if acc > 0.999:
        break
graph(9, layer(x_train))
plt.show()




