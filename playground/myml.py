import numpy as np
import numba
import math
import functools


class Layer:
    def input_shape(self):
        raise NotImplementedError("missing input_shape method")

    def output_shape(self):
        raise NotImplementedError("missing output_shape method")

    def num_params(self):
        return 0

    def apply(self, params, x):
        """Return the output of this layer, given the `params` and the input `x`."""
        raise NotImplementedError("missing apply method")


class InputLayer(Layer):
    """Layer that does nothing but specify the shape of the data coming in."""

    def __init__(self, shape):
        self.shape = shape

    def input_shape(self):
        return self.shape

    def output_shape(self):
        return self.shape

    def apply(self, _params, x):
        return x

    def derivatives(self, _params, _x, dz, _out):
        return dz


class FlattenLayer(Layer):
    """Reshape inputs to matrix form.

    This layer takes inputs with example-index as the first axis and any number of other axes.
    It flattens each example into a row, leaving a 2D matrix.

    That is, the actual input shape is `(num_examples, *example_shape)`
    and the output shape is `(num_examples, product(example_shape))`.
    """

    def __init__(self, example_shape):
        self.example_shape = example_shape

    def input_shape(self):
        return ('N', *self.example_shape)

    def output_shape(self):
        prd = functools.reduce(lambda a, b: a * b, self.example_shape, 1)
        return ('N', prd)

    def apply(self, _params, x):
        n = x.shape[0]
        if x.shape[1:] != self.example_shape:
            sizes = repr(self.example_shape)[1:-1]
            raise ValueError(f"Expected shape (N, {sizes}), got {x.shape!r}")
        return x.reshape((n, -1))

    def derivatives(self, _params, x, dz, _out):
        return dz.reshape((-1,) + self.example_shape)


class LinearLayer(Layer):
    """ A dense layer with a matrix of weights. """

    def __init__(self, shape):
        self.shape = shape

    def input_shape(self):
        return ('N', self.shape[0])

    def output_shape(self):
        return ('N', self.shape[1])

    def num_params(self):
        ni, no = self.shape
        return (ni + 1) * no

    def apply(self, params, x):
        ni, no = self.shape
        assert len(x.shape) == 2
        assert x.shape[1] == ni
        w = params[:ni * no].reshape((ni, no))
        b = params[ni * no:].reshape((1, no))
        return x @ w + b

    def derivatives(self, params, x, dz, out):
        """Given x and ∂L/∂z at x, compute partial derivatives ∂L/∂x, ∂L/∂w, and ∂L/∂b.

        Store ∂L/∂w and ∂L/∂b in `out`, a 1D vector of derivatives. Return ∂L/∂x.

        A step in backpropagation.

        dz[r,c] is the partial derivative of loss with respect to z[r,c]; that
        is, the partial derivative of the *rest* of the pipeline with respect to
        output r at training sample x[:,c].
        """

        ni, no = self.shape # number of outputs, inputs
        n = x.shape[0] # number of training samples
        assert x.shape == (n, ni)

        w = params[:ni * no].reshape((ni, no))

        assert dz.shape == (n, no)
        db = np.sum(dz, axis=0, keepdims=True)
        assert db.shape == (1, no)
        dw = x.T @ dz
        assert dw.shape == (ni, no)
        out[:ni * no] = dw.ravel()
        out[ni * no:] = db.ravel()

        dx = dz @ w.T
        assert dx.shape == x.shape
        return dx


class ActivationLayer(Layer):
    """ Applies the same nonlinear activation function to all outputs. """

    def input_shape(self):
        return 'X'

    def output_shape(self):
        return 'X'

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


class Add:
    def __init__(self, *args):
        self.args = args

    def __str__(self):
        expr = ' + '.join(str(a) for a in self.args)
        return f"({expr})"


class DivRoundUp:
    def __init__(self, n, d):
        self.n = n
        self.d = d

    def __str__(self):
        return f"({self.n} / {self.d} rounding up)"


def apply_shape(actual_shape, layer):
    input_shape = layer.input_shape()
    # maps layer's variables to shape-expressions in terms of actual_shape's variables.
    bindings = {}

    def unify(expr, pattern):
        if isinstance(pattern, int):
            if pattern != expr:
                raise ValueError(f"shape mismatch: expected {pattern}, previous layer had {expr}")
        elif isinstance(pattern, str):
            assert pattern.isalpha()
            if pattern in bindings:
                if bindings[pattern] != expr:
                    raise ValueError(f"error matching {actual_shape} to {layer.__class__.__name__} with input_shape {expected_shape}")
            else:
                bindings[pattern] = expr
        elif isinstance(pattern, tuple):
            if isinstance(expr, tuple):
                if len(pattern) != len(expr):
                    raise ValueError(f"shape mismatch: expected {pattern}, previous layer had {expr} (different number of dimensions)")
                for e, p in zip(expr, pattern):
                    unify(e, p)
        else:
            raise ValueError(f"unexpected input_shape {pattern}")

    def eval_expr(expr):
        if isinstance(expr, int):
            return expr
        elif isinstance(expr, str):
            return bindings[expr]
        elif isinstance(expr, tuple):
            return tuple(eval_expr(x) for x in expr)
        elif isinstance(expr, Add):
            vals = [eval_expr(x) for x in expr.args]
            if all(isinstance(v, int) for v in vals):
                return sum(vals)
            else:
                return Add(*vals)
        elif isinstance(expr, DivRoundUp):
            n, d = eval_expr(expr.n), eval_expr(expr.d)
            if isinstance(n, int) and isinstance(d, int):
                return (n + d - 1) // d
            else:
                return DivRoundUp(n, d)
        else:
            raise ValueError(f"unexpected type of output_shape {expr!r} ({expr.__class__.__name__})")

    unify(actual_shape, layer.input_shape())
    out = eval_expr(layer.output_shape())
    return out


class Sequence(Layer):
    def __init__(self, layers):
        self.layers = layers
        prev = self.layers[0]
        shape = prev.output_shape()
        for layer in self.layers[1:]:
            try:
                shape = apply_shape(shape, layer)
            except ValueError as exc:
                raise ValueError(
                    f"error matching output shape {shape!r} of layer {prev!r} "
                    f"to input shape {layer.input_shape()!r} of layer {layer!r}"
                ) from exc
            prev = layer
        self._out_shape = shape

    def describe(self):
        print('----')
        shape = self.input_shape()
        print("    shape:", shape)
        for layer in self.layers:
            if hasattr(layer, 'describe'):
                layer.describe()
            else:
                print(f"  {layer.__class__.__name__} ({layer.num_params()} params)")
                shape = apply_shape(shape, layer)
                print("    shape:", shape)
        print('----')

    def input_shape(self):
        return self.layers[0].input_shape()

    def output_shape(self):
        return self._out_shape

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

    def input_shape(self):
        return self.inner.input_shape()

    def output_shape(self):
        return self.inner.output_shape()

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
        return ex / np.sum(ex, axis=1, keepdims=True)

    def input_shape(self):
        return ('N', 'X')

    def output_shape(self):
        return ('N', 'X')

    def derivatives(self, _params, x, dz, _out):
        ex = np.exp(x)
        sum_ex = np.sum(ex, axis=1, keepdims=True)

        # Cell (i,j) of the result should be
        #     sum(∂loss/∂z[i,k] * ∂z[i,k]/∂ex[i,j] * ∂ex[i,j]/∂x[i,j]
        #         for k in range(no))
        #   = sum(dz[i,k]
        #         * ((sum_ex[i,1] if j == k else 0) - ex[i,k]) / sum_ex[i,1]**2
        #         * ex[i,j]
        #         for k in range(no))
        #   = (ex[i,j] / sum_ex[i,1]**2)
        #     * sum(dz[i,k] * ((sum_ex[i,1] if i == k else 0) - ex[i,k])
        #           for k in range(no))
        #   = (ex[i,j] / sum_ex[i,1]**2)
        #     * (dz[i,j] * sum_ex[i,1]
        #        - sum(dz[i,k] * ex[i,k] for k in range(no)))

        return (ex / sum_ex ** 2) * (dz * sum_ex - np.sum(dz * ex, axis=1, keepdims=True))


class LogisticLoss:
    """Loss function for logistic regression. AKA binary cross-entropy."""

    def loss(self, y, yh):
        return np.mean(-np.log(np.where(y == 0, 1 - yh, yh)))

    def deriv(self, y, yh):
        """Partial derivative of loss with respect to yh."""
        # When y == 0, we want the derivative of -log(1-yh)/n, or 1/(n*(1-yh)).
        # When y == 1, we want the derivative of -log(yh)/n, or -1/(n*yh).
        assert y.shape == yh.shape
        n, no = y.shape
        assert no == 1
        return 1.0 / (n * np.where(y == 0, 1 - yh, -yh))


class CategoricalCrossEntropyLoss:
    def loss(self, y, yh):
        n, c = yh.shape  # number of examples, number of categories
        assert y.shape == (n,)
        assert y.dtype.kind in ('i', 'u')
        assert np.all((y >= 0) & (y < c))
        assert np.all((yh >= 0.0) & (yh <= 1.0))
        return np.mean(-np.log(yh[np.arange(n), y]))

    def deriv(self, y, yh):
        ## There is a better way to say this but I don't remember it.
        n, c = yh.shape
        assert y.shape == (n,)
        row = np.arange(c)[np.newaxis, :]
        assert row.shape == (1, c)
        mask = y[:, np.newaxis] == row
        assert mask.shape == (n, c)
        return np.where(mask, -1.0 / (n * yh), 0)

    def accuracy(self, y, yh):
        predictions = np.argmax(yh, axis=1)
        return np.mean(predictions == y)


# Note: We assume throughout that indexes into images are like
# `images[image_index, y, x, channel]`.
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
    dk = np.zeros((zc, kh, kw, xc), dtype=dz.dtype)
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
    dx = np.zeros((n, xh, xw, xc), dtype=dz.dtype)
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


njit = numba.njit(cache=True, parallel=True, fastmath=True)
conv2d_impl = njit(conv2d_impl)
conv2d_dk = njit(conv2d_dk)
conv2d_dx = njit(conv2d_dx)


class Conv2DValidLayer(Layer):
    """2D convolution with no padding.

    In addition to the convolution kernel, each output channel gets a bias, a
    constant added to each pixel of that channel.
    """

    def __init__(self, kernel_shape):
        """kernel_shape must be (num_output_channels, height, width, num_img_channels).

        For example, to use 64 different 3x3 kernels on a grayscale image,
        use `(64, 3, 3, 1)`.
        """
        assert len(kernel_shape) == 4
        self.kernel_shape = kernel_shape

    def input_shape(self):
        return ('N', 'H', 'W', self.kernel_shape[3])

    def output_shape(self):
        oc, kh, kw, _ic = self.kernel_shape
        return ('N', Add('H', 1 - kh) , Add('W', 1 - kw), oc)

    def num_params(self):
        oc, kh, kw, ic = self.kernel_shape
        return oc * kh * kw * ic + oc

    def apply(self, params, x):
        # If this assertion fails, reshape the input images with something like
        # `images[..., np.newaxis]`.
        assert len(x.shape) == 4, "input must have shape (N, height, width, channels)"
        oc = self.kernel_shape[0]
        kernel = params[:-oc].reshape(self.kernel_shape)
        bias = params[-oc:]
        return conv2d_impl(x, kernel) + bias

    def derivatives(self, params, x, dz, out):
        oc, kh, kw, ic = self.kernel_shape
        out[:-oc] = conv2d_dk(x, dz).reshape((oc * kh * kw * ic,))
        out[-oc:] = dz.sum(axis=2).sum(axis=1).sum(axis=0)
        kernel = params[:-oc].reshape(self.kernel_shape)
        return conv2d_dx(kernel, dz)


class Pad2DLayer(Layer):
    """Add padding around each 2D input."""

    def __init__(self, y, x):
        """Make a padding layer.

        y - int - number of cells to add at the start and end of each column
        x - int - the same but for rows

        The input shape of the layer is (num_images, height, width, num_channels).
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
        _oc, kh, kw, ic = kernel_shape
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


class MaxPooling2DLayer(Layer):
    """ Max pooling operation for image data.

    The input shape is (num_images, height, width, num_channels).
    """
    def __init__(self, size=2):
        self.size = 2

    def input_shape(self):
        return ('N', 'H', 'W', 'C')

    def output_shape(self):
        return ('N', DivRoundUp('H', self.size), DivRoundUp('W', self.size), 'C')

    def apply(self, _params, x):
        ni, h, w, nc = x.shape
        pad_y = (-h) % self.size
        pad_x = (-w) % self.size
        if pad_x or pad_y:
            x = np.pad(x, ((0, 0), (0, pad_y), (0, pad_x), (0, 0)))
            w += pad_x
            h += pad_y
        out_w = w // self.size
        out_h = h // self.size
        x_grouped_1 = x.reshape((ni, out_h, self.size, w, nc))
        x_chosen_1 = np.amax(x_grouped_1, 2)
        x_grouped_2 = x_chosen_1.reshape((ni, out_h, out_w, self.size, nc))
        x_chosen_2 = np.amax(x_grouped_2, 3)
        return x_chosen_2

    def derivatives(self, _params, x, dz, _out):
        _ni, h, w, _nc = x.shape
        z = self.apply(_params, x)
        # ∂z/∂x is 1 exactly where an element's value is equal to the max,
        # and 0 elsewhere.
        m = np.repeat(np.repeat(z, self.size, axis=1), self.size, axis=2)[:,:h,:w,:]
        dz_stretched = np.repeat(np.repeat(dz, self.size, axis=1), self.size, axis=2)[:,:h,:w,:]
        return np.where(x == m, 1.0, 0.0) * dz_stretched


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
        self.params = 0.1 * rng.standard_normal((nparams,), dtype=np.float64)
        self.loss = loss
        self.learning_rate = 0.25
        self.last_loss = None
        self.last_params = None
        self.last_gradient = None

    def describe(self):
        self.seq.describe()

    def apply(self, x):
        return self.seq.apply(self.params, x)

    def train(self, x_train, y_train, rate):
        yh = self.seq.apply(self.params, x_train)
        loss = self.loss.loss(y_train, yh)

        accuracy = self.loss.accuracy(y_train, yh)

        dyh = self.loss.deriv(y_train, yh)
        dp = np.zeros((self.seq.num_params(),), dtype=np.float64)
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
        ## l = self.learning_rate
        ## print(f"loss={loss:.4f} accuracy={accuracy:.4f} λ={l}")

        self.last_accuracy = accuracy
        self.last_loss = loss
        ## self.last_params = self.params
        self.params = self.params - rate * dp
        ## self.last_gradient = dp

    def train_epochs(self, epochs):
        epochs = list(epochs)
        pairs_per_epoch = 0
        progress = 0.0

        for i, epoch in enumerate(epochs):
            print(f"epoch {i} - \x1b[s", end="", flush=True)
            n_total = 0
            loss_total = 0
            accuracy_total = 0

            for j, (x, y) in enumerate(epoch):
                n = len(x)
                n_total += n
                if i > 0:
                    batch_progress = min(n_total, pairs_per_epoch) / pairs_per_epoch
                    progress = ((i - 1) + batch_progress) / (len(epochs) - 1)
                self.train(x, y, self.learning_rate * (1.0 - progress))

                if n > 0:
                    loss_total += self.last_loss * n
                    accuracy_total += self.last_accuracy * n
                    loss = loss_total / n_total
                    accuracy = accuracy_total / n_total
                    print("\x1b[u" + 78 * " " + "\x1b[u"
                          + self._progress_bar(progress)
                          + f" loss={loss:.4f} accuracy={accuracy:.4f}",
                          end="", flush=True)

            if i == 0:
                pairs_per_epoch = n_total

            print()

    def _progress_bar(self, progress):
        LEN = 40
        n = int(LEN * min(progress, 1.0))
        return '[{}{}]'.format(n * '#', (LEN - n) * ' ')
