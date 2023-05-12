import numpy as np
from numpy.random import default_rng

import myml


def test_layer(rng, layer, input_shape):
    n = layer.num_params()
    params = rng.uniform(size=(n,))
    x = rng.uniform(size=input_shape)
    z = layer.apply(params, x)

    dz = rng.uniform(low=-0.1, high=0.1, size=z.shape)
    dp = np.zeros((n,))
    dx = layer.derivatives(params, x, dz, dp)

    e = 1e-6

    for i in range(n):
        # check accuracy of derivative at parameter i
        saved = params[i]
        params[i] = saved - e
        z_minus = layer.apply(params, x)
        params[i] = saved + e
        z_plus = layer.apply(params, x)
        params[i] = saved

        claimed = dp[i]
        measured = np.sum((z_plus - z_minus) / (2 * e) * dz)
        error = (claimed - measured) / measured
        assert abs(error) < 1e-5

    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        # check accuracy of derivative at input i
        saved = x[i]
        x[i] = saved - e
        z_minus = layer.apply(params, x)
        x[i] = saved + e
        z_plus = layer.apply(params, x)
        x[i] = saved

        claimed = dx[i]
        measured = np.sum((z_plus - z_minus) / (2 * e) * dz)
        error = (claimed - measured) / measured
        assert abs(error) < 1e-5


rng = default_rng()
test_layer(rng, myml.LinearLayer((1, 1)), (1, 1))
test_layer(rng, myml.LinearLayer((3, 5)), (5, 1))
test_layer(rng, myml.Conv2DValidLayer((2, 3, 3, 1)), (1, 4, 6, 1))
test_layer(rng, myml.Conv2DValidLayer((2, 3, 3, 3)), (2, 4, 6, 3))

