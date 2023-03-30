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

    def derivatives(self, _params, x, dz, _out):
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
        is, the partial derivative of the *rest* of the pipeline with respect to
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
    """Rectified linear unit activation function."""

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
        x = np.where(x > 400, 400, x) # avoid warning
        ex = np.exp(x)
        return ex / np.sum(ex, axis=0)

    def derivatives(self, _params, x, dz, _out):
        ex = np.exp(x)
        sum_ex = np.sum(ex, axis=0, keepdims=True)

        # Cell (i,j) of the result should be
        #     sum(∂loss/∂z[k,j] * ∂z[k,j]/∂ex[i,j] * ∂ex[i,j]/∂x[i,j]
        #         for k in range(no))
        #   = sum(dz[k,j]
        #         * ((sum_ex[1,j] if i == k else 0) - ex[k,j]) / sum_ex[1,j]**2
        #         * ex[i,j]
        #         for k in range(no))
        #   = (ex[i,j] / sum_ex[1,j]**2)
        #     * sum(dz[k,j] * ((sum_ex[1,j] if i == k else 0) - ex[k,j])
        #           for k in range(no))
        #   = (ex[i,j] / sum_ex[1,j]**2)
        #     * (dz[i,j] * sum_ex[1,j]
        #        - sum(dz[k,j] * ex[k,j] for k in range(no)))

        return (ex / sum_ex ** 2) * (dz * sum_ex - np.sum(dz * ex, axis=0, keepdims=True))


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


class CategoricalCrossEntropyLoss:
    def loss(self, y, yh):
        c, n = yh.shape  # number of categories, number of examples
        assert y.shape == (n,)
        assert y.dtype.kind in ('i', 'u')
        assert np.all((y >= 0) & (y < c))
        assert np.all((yh >= 0.0) & (yh <= 1.0))
        return np.mean(-np.log(yh[y, np.arange(n)]))

    def deriv(self, y, yh):
        ## There is a better way to say this but I don't remember it.
        c, n = yh.shape
        assert y.shape == (n,)
        column = np.arange(c)[:, np.newaxis]
        assert column.shape == (c, 1)
        mask = y[np.newaxis, :] == column
        assert mask.shape == (c, n)
        return np.where(mask, -1.0 / (n * yh), 0)

    def accuracy(self, y, yh):
        predictions = np.argmax(yh, axis=0)
        return np.mean(predictions == y)


def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between(u, v):
    u = unit_vector(u)
    v = unit_vector(v)
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))


class Model:
    def __init__(self, rng, seq, loss):
        self.seq = seq
        nparams = seq.num_params()
        print(f"creating model with {nparams} parameters")
        self.params = 0.02 * rng.standard_normal((nparams,))
        self.loss = loss
        self.learning_rate = 0.25
        self.last_loss = None
        self.last_params = None
        self.last_gradient = None

    def apply(self, x):
        return self.seq.apply(self.params, x)

    def train(self, x_train, y_train):
        yh = self.seq.apply(self.params, x_train)
        loss = self.loss.loss(y_train, yh)

        accuracy = self.loss.accuracy(y_train, yh)

        dyh = self.loss.deriv(y_train, yh)
        dp = np.zeros((self.seq.num_params(),))
        _ = self.seq.derivatives(self.params, x_train, dyh, dp)

        if np.all(dp == 0.0):
            print("gradient is 0")
            return

        ## dp = unit_vector(dp)
        ## if self.last_gradient is not None:
        ##     a = angle_between(self.last_gradient, dp)
        ##     if a > math.pi / 72:
        ##         print("slowing down")
        ##         self.learning_rate *= 1/11
        ##         if loss >= self.last_loss:
        ##             # actually revert params and gradient
        ##             self.params = self.last_params
        ##             dp = self.last_gradient
        ##     else:
        ##         self.learning_rate *= 1.25
        l = self.learning_rate

        print(f"loss={loss:.4f} accuracy={accuracy:.4f} λ={l}")

        ## self.last_loss = loss
        ## self.last_params = self.params
        self.params = self.params - self.learning_rate * dp
        ## self.last_gradient = dp
