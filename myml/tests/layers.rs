//! Tests that check for consistency between `apply` and `derivatives`.

use ndarray::prelude::*;
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use myml::*;

fn test_layer<L, D>(layer: L, input_shape: D)
where
    D: IntoDimension,
    L: Layer<D::Dim>,
{
    let input_shape = input_shape.into_dimension();
    let error_limit = 3e-3;

    let n = layer.num_params();
    let mut params = Array::random(n, Uniform::new(0.0, 1.0));
    println!("testing layer {layer:#?} with parameters {params:#?}");
    let mut x = Array::random(input_shape.clone(), Uniform::new(0.0, 1.0));
    println!("input x: {x:#?}");
    let z = layer.apply(params.view(), x.view());
    println!("output z: {z:#?}");

    let dz = Array::random(z.raw_dim(), Uniform::new(-0.1, 0.1));
    println!("dz: {dz:#?}");
    let mut dp = Array::zeros(n);
    let dx = layer.derivatives(params.view(), x.view(), dz.view(), dp.view_mut());
    println!("dx: {dx:#?}");

    let h = 3e-4;

    fn err(claimed: f32, measured: f32) -> f32 {
        let d = measured.abs().max(1e-3);
        (claimed - measured).abs() / d
    }

    for i in 0..n {
        // check accuracy of derivative at parameter params[i]
        let saved = params[i];
        params[i] = saved - h;
        let z_minus = layer.apply(params.view(), x.view());
        params[i] = saved + h;
        let z_plus = layer.apply(params.view(), x.view());
        params[i] = saved;

        let claimed = dp[i];
        let measured = ((z_plus - z_minus) * (1.0 / (2.0 * h)) * &dz).sum();

        let error = err(claimed, measured);
        assert!(
            error <= error_limit,
            "parameter {i} computed derivative = {claimed}, measured = {measured}, error = {error}, limit = {error_limit}"
        );
    }

    for i in ndarray::indices(input_shape) {
        // check accuracy of derivative at input x[i]
        let i = i.into_dimension();
        let saved = x[i.clone()];
        x[i.clone()] = saved - h;
        let z_minus = layer.apply(params.view(), x.view());
        x[i.clone()] = saved + h;
        let z_plus = layer.apply(params.view(), x.view());
        x[i.clone()] = saved;

        let claimed = dx[i.clone()];
        let measured = ((z_plus - z_minus) * (1.0 / (2.0 * h)) * &dz).sum();

        let error = err(claimed, measured);
        assert!(
            error <= error_limit,
            "input element {i:?} computed derivative = {claimed}, measured = {measured}, error = {error}, limit = {error_limit}"
        );
    }
}

#[test]
fn test_layer_consistency() {
    test_layer(layers::LinearLayer::new(1, 1), (1, 1));
    test_layer(layers::LinearLayer::new(5, 3), (1, 5));

    //test_layer(layers::LinearLayer::new(10, 10).relu(), (2, 10));
    //test_layer(layers::LinearLayer::new(4, 2).softmax(), (3, 4));
    //test_layer(layers::LinearLayer::new(30, 10).relu().linear(10, 4).softmax(), (2, 30));

    // test_layer(rng, myml.Conv2DValidLayer((2, 3, 3, 1)), (1, 4, 6, 1))
    // test_layer(rng, myml.Conv2DValidLayer((2, 3, 3, 3)), (2, 4, 6, 3))
    // test_layer(rng, myml.MaxPooling2DLayer(2), (1, 6, 6, 3))
    // test_layer(rng, myml.MaxPooling2DLayer(2), (1, 3, 3, 3))
    // test_layer(rng, myml.MaxPooling2DLayer(3), (1, 4, 5, 3))
}
