#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import matplotlib
import matplotlib.pyplot as plt
import math


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


class Layer:
    def len_params(self):
        return 0


class LinearLayer(Layer):
    """ A dense layer with a matrix of weights. """
    
    def __init__(self, rng, shape):
        num_outputs, num_inputs = shape
        self.w = 0.02 * rng.standard_normal(shape)
        self.b = 0.02 * rng.standard_normal((num_outputs, 1))

    def len_params(self):
        no, ni = self.w.shape
        return no * (ni + 1)

    def get_param(self, i):
        no, ni = self.w.shape
        if i < no * ni:
            return self.w[i // ni, i % ni]
        else:
            return self.b[i - no * ni, 0]

    def set_param(self, i, val):
        no, ni = self.w.shape
        if i < no * ni:
            self.w[i // ni, i % ni] = val
        else:
            self.b[i - no * ni, 0] = val

    def apply(self, x):
        return self.w @ x + self.b

    def derivatives(self, x, dz):
        """Given x and ∂L/∂z at x, return the partial derivatives (∂L/∂x, (∂L/∂w, ∂L/∂b)).

        Backpropagation.

        dz[r,c] is the partial derivative of loss with respect to z[r,c]; that
        is the partial derivative of the *rest* of the pipeline with respect to
        output r at training sample x[:,c].
        """

        no, ni = self.w.shape # number of outputs, inputs
        n = x.shape[1] # number of training samples
        assert x.shape == (ni, n)

        assert dz.shape == (no, n)
        db = np.sum(dz, axis=1, keepdims=True)
        assert db.shape == (no, 1)
        dw = dz @ x.T
        assert dw.shape == (no, ni)

        dx = self.w.T @ dz
        assert dx.shape == x.shape
        return dx, (dw, db)

    def learn(self, learning_rate, derivs):
        dw, db = derivs
        self.b -= learning_rate * db
        self.w -= learning_rate * dw


class ActivationLayer(Layer):
    """ Applies the same nonlinear activation function to all outputs. """
    def apply(self, x):
        return self.f(x)

    def derivatives(self, x, dz):
        return (self.df(x) * dz, None)

    def learn(self, _learning_rate, _derivs):
        # No parameters to learn.
        pass


def sigmoid(v):
    """The logistic function, a handy symmetric, s-shaped function."""
    v = np.where(v > -700, v, -700) # avoid overflow warnings
    return 1 / (1 + np.exp(-v))


class SigmoidLayer(ActivationLayer):
    """ The logistic function """

    def f(self, x):
        return sigmoid(x)

    def df(self, x):
        y = sigmoid(x)
        return y * (1 - y)


class ReluLayer(ActivationLayer):
    """Rectivied linear unit activation function."""

    def f(self, x):
        return np.where(x >= 0, x, 0)

    def df(self, x):
        return np.where(x >= 0, 1, 0)


class Sequence(Layer):
    def __init__(self, layers):
        self.layers = layers

    def len_params(self):
        return sum(l.len_params() for l in self.layers)

    def get_param(self, i):
        for layer in self.layers:
            n = layer.len_params()
            if i < n:
                return layer.get_param(i)
            i -= n
        raise IndexError("parameter out of range")

    def set_param(self, i, value):
        for layer in self.layers:
            n = layer.len_params()
            if i < n:
                return layer.set_param(i, value)
            i -= n
        raise IndexError("parameter out of range")

    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x)
        return x

    def derivatives(self, x, dz):
        # redo all the work of apply :\
        values = []
        for layer in self.layers:
            values.append(x)
            x = layer.apply(x)

        dparams = []
        for layer, x in zip(reversed(self.layers), reversed(values)):
            dz, dparams_i = layer.derivatives(x, dz)
            dparams.append(dparams_i)

        return dz, dparams[::-1]

    def learn(self, learning_rate, dparams):
        for layer, dparams_i in zip(self.layers, dparams):
            layer.learn(learning_rate, dparams_i)


class LogisticLoss:
    """Loss function for logistic regression. AKA binary cross-entropy."""

    def loss(self, y, yh):
        return np.mean(-np.log(np.where(y == 0, 1 - yh, yh)))

    def deriv(self, y, yh):

        """Partial derivative of loss with respect to yh."""
        # When y == 0, we want the derivative of -log(1-yh)/n, or 1/(n*(1-yh)).
        # When y == 1, we want the derivative of -log(yh)/n, or -1/(n*yh).
        assert y.shape == yh.shape
        no, n = y.shape
        assert no == 1
        return 1.0 / (n * np.where(y == 0, 1 - yh, -yh))


def unit_vector(v):
    (dw, db), _ = v # HACK
    v = np.array([dw[0][0], dw[0][1], db[0][0]], dtype=float) # HACK
    return v / np.linalg.norm(v)


def angle_between(u, v):
    u = unit_vector(u)
    v = unit_vector(v)
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

class Model:
    def __init__(self, seq, loss):
        self.seq = seq
        self.loss = loss
        self.last_gradient = None

    def train(self, x_train, y_train, learning_rate):
        yh = self.seq.apply(x_train)
        loss = self.loss.loss(y_train, yh)
        print("loss:", loss)
        dyh = self.loss.deriv(y_train, yh)
        _, dp = self.seq.derivatives(x_train, dyh)
        if self.last_gradient is not None:
            a = angle_between(self.last_gradient, dp)
            print("change in direction:", a)
            if a > math.pi/72:
                learning_rate /= 1.01
            else:
                learning_rate *= 1.01
        self.seq.learn(learning_rate, dp)
        self.last_gradient = dp


##    def train(self, x, y, learn_rate):
##        db, dw = self.derivatives(x, y)
##
##        step_size = np.linalg.norm(learn_rate * np.concatenate((dw.flatten(), db.flatten())))
##        if 0 < step_size < 0.01:
##            learn_rate *= 0.01 / step_size
##        self.b -= learn_rate * db
##        self.w -= learn_rate * dw

##     def parameters(self):
##         """Return a 1D array of all parameters."""
##         no, ni = self.w.shape
##         return np.concatenate((self.w.flatten(), self.b.flatten()))
## 
##     def check_derivs(self, x, y, dw, db):
##         no, ni = self.w.shape
## 
##         def loss(params):
##             w = params[0:no * ni].reshape(self.w.shape)
##             b = params[no * ni:].reshape(self.b.shape)
##             a = sigmoid(w @ x + b)
##             q = np.where(y == 0, 1 - a, a)
##             return np.mean(-np.log(q))
## 
##         def approx_derivs(params):
##             EPSILON = 1.0e-7
##             derivs = []
##             for i in range(len(params)):
##                 orig = params[i]
##                 h = EPSILON * max(abs(orig), EPSILON)
##                 params[i] = orig - h
##                 la = loss(params)
##                 params[i] = orig + h
##                 lb = loss(params)
##                 params[i] = orig
##                 derivs.append((lb - la) / (2 * h))
##             return np.array(derivs)
## 
##         params = self.parameters()
##         dapprox = approx_derivs(params)
##         derivs = np.concatenate((dw.flatten(), db.flatten()))
##         e = np.linalg.norm(derivs - dapprox)
##         if e > 1.0e-5 * np.linalg.norm(derivs):
##             print("WARNING - derivatives may be wrong")
##             print("  computed derivs:", derivs)
##             print("  numerical derivs:", dapprox)
##             print("  difference:", e)

def main():
    rng = default_rng()
    (x_train, y_train), (x_test, y_test) = generate_data(rng, 100, 100)

    model = Model(
        Sequence([
            LinearLayer(rng, (1, 2)),
            SigmoidLayer(),
        ]),
        LogisticLoss(),
    )

    #layer = DenseLayer(rng, (1, 2))

    def graph(i):
        plt.subplot(2, 5, i + 1)
        yh = model.seq.apply(x_train)
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
        model.train(x_train, y_train, learn_rate)

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
