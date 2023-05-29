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
    arrays of shape (n, 2); and both _y values are arrays of shape (n, 1)
    where each value is 0 or 1.
    """
    n = (ntraining + ntesting + 1) // 2
    a = rng.standard_normal(size=(NDIMS, n))
    b = 5.0 + rng.standard_normal(size=(NDIMS, n))

    a = np.r_[a, np.zeros((1, n))]
    b = np.r_[b, 1 + np.zeros((1, n))]

    points = np.c_[a, b]
    rng.shuffle(points, axis=1)

    def split_xy(data):
        return data[0:2], data[2:3]
    return split_xy(points[:,:ntraining]), split_xy(points[:,ntraining:ntraining + ntesting])


def sigmoid(v):
    """A handy symmetric, s-shaped function. Used in our model."""
    v = np.where(v > -700, v, -700) # avoid overflow warnings
    return 1 / (1 + np.exp(-v))


class DenseLayer:
    """ A dense layer of perceptrons, like tensorflow.keras.layers.Dense. """

    def __init__(self, rng, shape):
        num_outputs, num_inputs = shape
        self.w = 0.02 * rng.standard_normal(shape)
        self.b = 0.02 * rng.standard_normal((num_outputs, 1))

    def __call__(self, x):
        return sigmoid(self.w @ x + self.b)

    def accuracy(self, x, y):
        """Percentage of the examples in x where we correctly predict y.

        Each point is counted as either right or wrong.
        """
        yh = self(x)
        return np.mean((yh < 1/2) == (y < 1/2))

    def loss(self, x, y):
        """ The loss function is what training seeks to minimize. """
        yh = self(x)
        q = np.where(y == 0, 1 - yh, yh)
        return np.mean(-np.log(q))

    def train(self, x, y, learn_rate):
        db, dw = self.derivatives(x, y)

        step_size = np.linalg.norm(learn_rate * np.concatenate((dw.flatten(), db.flatten())))
        if 0 < step_size < 0.01:
            learn_rate *= 0.01 / step_size
        self.b -= learn_rate * db
        self.w -= learn_rate * dw

    def derivatives(self, x, y):
        """Return the partial derivatives ∂L/∂b and ∂L/∂w.

        That is, characteraze how tweaking each parameter of the model would
        affect the loss.

        Returns two arrays: the first, ∂L/∂w, is an array the same shape as
        self.w, and each element tells the partial derivative of the loss with
        respect to the corresponding element of w. The second, ∂L/∂b, is the
        same for self.b.
        """

        no, ni = self.w.shape # number of outputs, inputs
        n = x.shape[1] # number of training samples
        assert x.shape == (ni, n)
        assert y.shape == (no, n)

        # Run the model, computing the output yh and the loss, keeping all
        # intermediate values.
        z = self.w @ x + self.b
        assert z.shape == (no, n)
        yh = sigmoid(z)
        assert yh.shape == (no, n)
        if not np.all(np.isfinite(yh)):
            raise ValueError("non-finite output")
        q = np.where(y == 0, 1 - yh, yh)
        loss = np.mean(-np.log(q))  # scalar
        ## print("loss:", loss)
        if not np.isfinite(loss):
            raise ValueError("non-finite loss")

        # Now work backwards and compute the derivatives.
        # Following the convention, for ∂L/∂x we write `dx`.
        dq = 1 / n * -1 / q
        assert dq.shape == (no, n)
        dyh = dq * np.where(y == 0, -1, 1)
        assert dyh.shape == (no, n)
        dz = dyh * -1 / (1 + np.exp(-z))**2 * -np.exp(-z)
        assert dz.shape == (no, n)
        db = np.sum(dz, axis=1, keepdims=True)
        assert db.shape == (no, 1)
        dw = dz @ x.T
        assert dw.shape == (no, ni)

        return db, dw

    def parameters(self):
        """Return a 1D array of all parameters."""
        no, ni = self.w.shape
        return np.concatenate((self.w.flatten(), self.b.flatten()))

    def check_derivs(self, x, y, dw, db):
        no, ni = self.w.shape

        def loss(params):
            w = params[0:no * ni].reshape(self.w.shape)
            b = params[no * ni:].reshape(self.b.shape)
            a = sigmoid(w @ x + b)
            q = np.where(y == 0, 1 - a, a)
            return np.mean(-np.log(q))

        def approx_derivs(params):
            EPSILON = 1.0e-7
            derivs = []
            for i in range(len(params)):
                orig = params[i]
                h = EPSILON * max(abs(orig), EPSILON)
                params[i] = orig - h
                la = loss(params)
                params[i] = orig + h
                lb = loss(params)
                params[i] = orig
                derivs.append((lb - la) / (2 * h))
            return np.array(derivs)

        params = self.parameters()
        dapprox = approx_derivs(params)
        derivs = np.concatenate((dw.flatten(), db.flatten()))
        e = np.linalg.norm(derivs - dapprox)
        if e > 1.0e-5 * np.linalg.norm(derivs):
            print("WARNING - derivatives may be wrong")
            print("  computed derivs:", derivs)
            print("  numerical derivs:", dapprox)
            print("  difference:", e)

def main():
    rng = default_rng()
    (x_train, y_train), (x_test, y_test) = generate_data(rng, 800, 100)
    layer = DenseLayer(rng, (1, 2))

    def graph(i):
        plt.subplot(2, 5, i + 1)
        yh = layer(x_train)
        plt.scatter(x=x_train[0], y=x_train[1], c=yh)
        # superimpose the points we mispredict in red
        err_points = x_train[:, (yh[0] < 1/2) != (y_train[0] < 1/2)]
        plt.scatter(x=err_points[0], y=err_points[1], c='red', s=6)

    NROUNDS = 298
    decile = (NROUNDS - 1) // 9
    for i in range(NROUNDS):
        if i % decile == 0:
            graph(i // decile)
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
    graph(9)
    plt.show()


main()
