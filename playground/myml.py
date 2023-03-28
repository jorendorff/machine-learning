import numpy as np
import math


class Layer:
    def num_params(self):
        return 0


class FlattenLayer(Layer):
    """Reshape inputs to matrix form.

    This layer takes inputs with example-index as the first axis and any number of other axes.
    It reshapes and transposes the data into a 2D matrix where each column is one example.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def apply(self, _params, x):
        n = x.shape[0]
        assert x.shape[1:] == self.input_shape
        y = x.reshape((n, -1)).T
        return y

    def derivatives(self, x, dz, _out):
        return dz.reshape((-1,) + self.input_shape)


class LinearLayer(Layer):
    """ A dense layer with a matrix of weights. """

    def __init__(self, shape):
        self.shape = shape

    def num_params(self):
        no, ni = self.shape
        return no * (ni + 1)

    def apply(self, params, x):
        no, ni = self.shape
        w = params[:no * ni].reshape((no, ni))
        b = params[no * ni:].reshape((no, 1))
        return w @ x + b

    def derivatives(self, params, x, dz, out):
        """Given x and ∂L/∂z at x, compute partial derivatives ∂L/∂x, ∂L/∂w, and ∂L/∂b.

        Store ∂L/∂w and ∂L/∂b in `out`, a 1D vector of derivatives. Return ∂L/∂x.

        Backpropagation.

        dz[r,c] is the partial derivative of loss with respect to z[r,c]; that
        is the partial derivative of the *rest* of the pipeline with respect to
        output r at training sample x[:,c].
        """

        no, ni = self.shape # number of outputs, inputs
        n = x.shape[1] # number of training samples
        assert x.shape == (ni, n)

        w = params[:no * ni].reshape((no, ni))

        assert dz.shape == (no, n)
        db = np.sum(dz, axis=1, keepdims=True)
        assert db.shape == (no, 1)
        dw = dz @ x.T
        assert dw.shape == (no, ni)
        out[:no * ni] = dw.ravel()
        out[no * ni:] = db.ravel()

        dx = w.T @ dz
        assert dx.shape == x.shape
        return dx


class ActivationLayer(Layer):
    """ Applies the same nonlinear activation function to all outputs. """
    def apply(self, _params, x):
        return self.f(x)

    def derivatives(self, _params, x, dz, _out):
        return self.df(x) * dz


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

    def num_params(self):
        return sum(l.num_params() for l in self.layers)

    def apply(self, params, x):
        i0 = 0
        for layer in self.layers:
            i1 = i0 + layer.num_params()
            x = layer.apply(params[i0:i1], x)
            i0 = i1
        return x

    def derivatives(self, params, x, dz, out):
        # redo all the work of apply :\
        values = []
        i0 = 0
        for layer in self.layers:
            values.append(x)
            i1 = i0 + layer.num_params()
            x = layer.apply(params[i0:i1], x)
            i0 = i1

        i1, = out.shape
        for layer, x in zip(reversed(self.layers), reversed(values)):
            i0 = i1 - layer.num_params()
            dz = layer.derivatives(params[i0:i1], x, dz, out[i0:i1])
            i1 = i0

        return dz


class SoftmaxLayer(Layer):
    def apply(self, _params, x):
        return np.mean(np.exp(x), axis=0, keepdims=True)

    def derivatives(self, x, dz, _out):
        # We have dL/dz. We seek dL/dx = dL/dz * dz/dx,
        # where dz/dx = d/dx (exp(x) / sum(exp(xi))).
        # By the quotient rule this is = exp(x) * (sum(exp(xi)) - exp(x)) / sum(exp(xi)) ** 2
        ex = np.exp(x)
        sum_ex = np.sum(ex, axis=0, keepdims=True)
        return dz * (ex * (sum_ex - ex) / sum_ex ** 2)


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


class CategoryCrossEntropyLoss:
    def loss(self, y, yh):
        raise NotImplementedError

    def deriv(self, y, yh):
        raise NotImplementedError


def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between(u, v):
    u = unit_vector(u)
    v = unit_vector(v)
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))


class Model:
    def __init__(self, rng, seq, loss):
        self.seq = seq
        self.params = 0.02 * rng.standard_normal((seq.num_params(),))
        self.loss = loss
        self.learning_rate = 0.05
        self.last_gradient = None

    def apply(self, x):
        return self.seq.apply(self.params, x)

    def train(self, x_train, y_train):
        yh = self.seq.apply(self.params, x_train)
        loss = self.loss.loss(y_train, yh)
        print("loss:", loss)
        dyh = self.loss.deriv(y_train, yh)
        dp = np.zeros((self.seq.num_params(),))
        _ = self.seq.derivatives(self.params, x_train, dyh, dp)
        if self.last_gradient is not None:
            a = angle_between(self.last_gradient, dp)
            print("change in direction:", a)
            if a > math.pi / 72:
                self.learning_rate /= 10
            elif self.learning_rate < 10:
                self.learning_rate *= 1.1
        print("learning rate:", self.learning_rate)
        self.params -= self.learning_rate * dp
        self.last_gradient = dp
