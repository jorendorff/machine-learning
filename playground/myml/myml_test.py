import numpy as np
from numpy.random import default_rng

import myml


def test_layer(rng, layer, input_shape, *, error_limit=1e-5):
    n = layer.num_params()
    params = rng.uniform(size=(n,))
    x = rng.uniform(size=input_shape)
    z = layer.apply(params, x)

    dz = rng.uniform(low=-0.1, high=0.1, size=z.shape)
    dp = np.zeros((n,))
    dx = layer.derivatives(params, x, dz, dp)

    h = 1e-7

    def err(claimed, measured):
        # How bad is the claimed value?
        denom = max(abs(measured), 1e-3)
        return abs((claimed - measured) / denom)

    for i in range(n):
        # check accuracy of derivative at parameter i
        saved = params[i]
        params[i] = saved - h
        z_minus = layer.apply(params, x)
        params[i] = saved + h
        z_plus = layer.apply(params, x)
        params[i] = saved

        claimed = dp[i]
        measured = np.sum((z_plus - z_minus) / (2 * h) * dz)

        error = err(claimed, measured)
        if error > error_limit:
            raise ValueError(f"parameter {i} computed derivative = {claimed}, measured = {measured}, error = {error}, limit = {error_limit}")

    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        # check accuracy of derivative at input i
        saved = x[i]
        x[i] = saved - h
        z_minus = layer.apply(params, x)
        x[i] = saved + h
        z_plus = layer.apply(params, x)
        x[i] = saved

        claimed = dx[i]
        measured = np.sum((z_plus - z_minus) / (2 * h) * dz)
        error = err(claimed, measured)
        if error > error_limit:
            raise ValueError(f"input element {i} computed derivative = {claimed}, measured = {measured}, error = {error}, limit = {error_limit}")


rng = default_rng()
test_layer(rng, myml.LinearLayer((1, 1)), (1, 1))
test_layer(rng, myml.LinearLayer((5, 3)), (1, 5))
test_layer(rng, myml.Conv2DValidLayer((2, 3, 3, 1)), (1, 4, 6, 1))
test_layer(rng, myml.Conv2DValidLayer((2, 3, 3, 3)), (2, 4, 6, 3))
test_layer(rng, myml.MaxPooling2DLayer(2), (1, 6, 6, 3))
test_layer(rng, myml.MaxPooling2DLayer(2), (1, 3, 3, 3))
test_layer(rng, myml.MaxPooling2DLayer(3), (1, 4, 5, 3))

test_layer(rng, myml.Sequence([
    myml.LinearLayer((10, 10)),
    myml.ReluLayer(),
]), (2, 10))

test_layer(rng, myml.Sequence([
    myml.LinearLayer((4, 2)),
    myml.SoftmaxLayer(),
]), (3, 4))

test_layer(rng, myml.Sequence([
    myml.LinearLayer((30, 10)),
    myml.ReluLayer(),
    myml.LinearLayer((10, 4)),
    myml.SoftmaxLayer(),
]), (2, 30))

test_layer(rng, myml.Sequence([
    myml.Conv2DValidLayer((32, 3, 3, 1)),
    myml.ReluLayer(),
    myml.MaxPooling2DLayer(),
]), (1, 28, 28, 1))

# check that conv2d actually does anything reasonable
c2d = myml.Conv2DValidLayer((2, 3, 3, 1))
assert c2d.num_params() == 2 * 3 * 3 + 2
out = c2d.apply(
    np.array([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,

        0.1, 0.2, 0.3,
        0.01, 0.02, 0.03,
        0.001, 0.002, 0.003,

        10.0, 20.0,
    ]),
    np.array([
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]).reshape((1, 3, 3, 1)),
)
assert str(out) == "[[[[12.   20.42]]]]"
