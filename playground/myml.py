import numpy as np
import numba
import math


class Layer:
    def num_params(self):
        return 0

    def apply(self, params, x):
        """Return the output of this layer, given the `params` and the input `x`.

        `x` is a NumPy ndarray where the *last* axis is the training mini-batch axis.
        For example, if the inputs are 640x480 images, and there are 10 of them in the
        current mini-batch, then `x.shape == (640, 480, 10)`.

        The output is a NumPy ndarray where the last axis is again the training
        mini-batch axis.
        """
        raise NotImplementedError("missing apply method")


class FlattenLayer(Layer):
    """Reshape inputs to matrix form.

    This layer takes inputs with example-index as the first axis and any number of other axes.
    It reshapes and transposes the data into a 2D matrix where each column is one example.

    That is, the input shape is `(num_examples, *input_shape)`
    and the output shape is `(product(input_shape), num_examples)`.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def apply(self, _params, x):
        n = x.shape[0]
        assert x.shape[1:] == self.input_shape
        y = x.reshape((n, -1)).T
        return y

    def derivatives(self, _params, x, dz, _out):
        return dz.T.reshape((-1,) + self.input_shape)


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


class Residual(Layer):
    def __init__(self, inner):
        self.inner = inner

    def num_params(self):
        return self.inner.num_params()

    def apply(self, params, x):
        return x + self.inner.apply(params, x)

    def derivatives(self, params, x, dz, out):
        dx = self.inner.derivatives(params, x, dz, out)
        return dx + dz


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


# Note: We assume throughout that indexes into images are like
# `images[y, x, channel, image_index]`.
#
# And for kernels: `kernel[output_channel, ky, kx, image_channel]`.
#
# Kernel sizes are (height, width) to match this (y, x) order.


def conv2d_impl(images, kernel):
    """ Convolve matrices in 2D as needed for a convolutional neural network.

    The shape of `images` is (img_width, img_height, num_img_channels, num_images).
    The shape of `kernel` is (num_output_channels, knl_width, knl_height, num_img_channels).
    Note that `num_img_channels` must agree.
    The output shape is (new_img_width, new_img_height, num_output_channels, num_images).

    Implementation is after <https://stackoverflow.com/a/64660822>.
    """

    # number of images in batch, height, width, number of channels
    xn, xh, xw, xc = images.shape
    # number of outputs, height, width, number of channels
    kn, kh, kw, kc = kernel.shape
    if kc != xc:
        raise ValueError(f"incompatible number of channels: images={xc}, kernel={kc}")

    ow = xw - kw + 1
    oh = xh - kh + 1
    out = np.zeros((xn, oh, ow, kn), dtype=images.dtype)
    for t in range(xn):
        for j in range(kh):
            for i in range(kw):
                for ic in range(xc):
                    for oc in range(kn):
                        for y in range(oh):
                            for x in range(ow):
                                out[t, y, x, oc] += \
                                    kernel[oc, j, i, ic] * images[t, y + j, x + i, ic]
    return out


def conv2d_dk(images, dz):
    """ Compute derivatives of loss with respect to Conv2D kernel parameters. """
    xn, xh, xw, xc = images.shape
    zn, zh, zw, zc = dz.shape
    if zn != xn:
        raise ValueError(f"incompatible number of samples: images={xn}, dz={zn}")
    kw = xw - zw + 1
    kh = xh - zh + 1
    dk = np.zeros((zc, kh, kw, xc))
    for t in range(xn):
        for ky in range(kh):
            for kx in range(kw):
                for ci in range(xc):
                    for co in range(zc):
                        for oy in range(zh):
                            for ox in range(zw):
                                dk[co, ky, kx, ci] += \
                                    images[t, oy + ky, ox + kx, ci] * dz[t, oy, ox, co]
    return dk


def conv2d_dx(kernel, dz):
    """Compute derivatives of loss with respect to Conv2D input image pixels."""
    kc, kh, kw, xc = kernel.shape
    n, zh, zw, zc = dz.shape
    if kc != zc:
        raise ValueError(f"incompatible number of channels: kernel={kc}, dz={zc}")
    xh = zh + kh - 1
    xw = zw + kw - 1
    dx = np.zeros((n, xh, xw, xc))
    for t in range(n):
        for ky in range(kh):
            for kx in range(kw):
                for ci in range(xc):
                    for co in range(zc):
                        for zy in range(zh):
                            for zx in range(zw):
                                dx[t, zy + ky, zx + kx, ci] += \
                                    kernel[co, ky, kx, ci] * dz[t, zy, zx, co]
    return dx


# njit = numba.njit(cache=True, parallel=True, fastmath=True)
# conv2d_impl = njit(conv2d_impl)
# conv2d_dk = njit(conv2d_dk)
# conv2d_dx = njit(conv2d_dx)


class Conv2DValidLayer(Layer):
    """2D convolution with no padding."""

    def __init__(self, kernel_shape):
        """kernel_shape must be (num_output_channels, height, width, num_img_channels).

        For example, to use 64 different 3x3 kernels on a grayscale image,
        use `(64, 3, 3, 1)`.
        """
        assert len(kernel_shape) == 4
        self.kernel_shape = kernel_shape

    def num_params(self):
        oc, kh, kw, ic = self.kernel_shape
        return oc * kh * kw * ic

    def apply(self, params, x):
        kernel = params.reshape(self.kernel_shape)
        return conv2d_impl(x, kernel)

    def derivatives(self, params, x, dz, out):
        out[:] = conv2d_dk(x, dz).reshape(out.shape)
        kernel = params.reshape(self.kernel_shape)
        return conv2d_dx(kernel, dz)


class Pad2DLayer(Layer):
    """Add padding around each 2D input."""

    def __init__(self, y, x):
        """Make a padding layer.

        y - int - number of cells to add at the start and end of each column
        x - int - the same but for rows

        The input shape of the layer is (height, width, num_channels, num_images).
        """
        self.pad_y = y
        self.pad_x = x

    def apply(self, _params, x):
        py = self.pad_y
        px = self.pad_x
        return numpy.pad(x, ((0, 0), (py, py), (px, px), (0, 0)))

    def derivatives(self, _params, _x, dz, _out):
        y = self.pad_y
        x = self.pad_x
        return dz[:,y:-y,x:-x,:]


class Conv2dSameLayer(Sequence):
    """2D convolution with zero-padding to keep the same size."""

    def __init__(self, kernel_shape):
        oc, kh, kw, ic = kernel_shape
        if kh % 2 != 1:
            raise ValueError(f"kernel height must be an odd number (got {kh})")
        if kw % 2 != 1:
            raise ValueError(f"kernel width must be an odd number (got {kw})")
        pad_x = (kw - 1) // 2
        pad_y = (kh - 1) // 2
        Sequence.__init__(self, [
            Pad2DLayer(pad_x, pad_y),
            Conv2DValidLayer(kernel_shape)
        ])


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

        #print(f"loss={loss:.4f} accuracy={accuracy:.4f} λ={l}")

        self.last_accuracy = accuracy
        self.last_loss = loss
        ## self.last_params = self.params
        self.params = self.params - self.learning_rate * dp
        ## self.last_gradient = dp

    def train_epochs(self, epochs):
        for i, epoch in enumerate(epochs):
            print(f"epoch {i} - \x1b[s", end="")
            n_total = 0
            loss_total = 0
            accuracy_total = 0
            for x, y in epoch:
                n = len(x)
                n_total += n
                self.train(x, y)

                if n > 0:
                    loss_total += self.last_loss * n
                    accuracy_total += self.last_accuracy * n
                    loss = loss_total / n_total
                    accuracy = accuracy_total / n_total
                    print("\x1b[u" + 50 * " " + "\x1b[u"
                          + f"loss={loss:.4f} accuracy={accuracy:.4f}",
                          end="")
            print()
