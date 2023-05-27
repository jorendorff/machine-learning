//! Tests that check for consistency between `apply` and `derivatives`.

use ndarray::prelude::*;
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use myml::*;

fn test_layer<L, D>(layer: L, input_shape: D, error_limit: f32)
where
    D: IntoDimension,
    L: Layer<D::Dim>,
{
    let input_shape = input_shape.into_dimension();
    let output_shape = layer.output_shape(input_shape.clone());

    let n = layer.num_params();
    let mut params = Array::random(n, Uniform::new(0.0, 1.0));
    let mut x = Array::random(input_shape.clone(), Uniform::new(0.0, 1.0));
    println!("testing layer {layer:#?} with parameters {params:#?}, input {x:#?}");
    //println!("input x: {x:#?}");
    let mut tmp = Array2::zeros(layer.hidden_activations_shape(input_shape.clone()));
    let mut z = Array::zeros(output_shape.clone());
    layer.apply(params.view(), x.view(), tmp.view_mut(), z.view_mut());
    //println!("output z: {z:#?}");

    let dz = Array::random(z.raw_dim(), Uniform::new(-0.1, 0.1));
    //println!("dz: {dz:#?}");
    let mut dp = Array::zeros(n);
    let dx = layer.derivatives(
        params.view(),
        x.view(),
        tmp.view(),
        dz.view(),
        dp.view_mut(),
    );
    //println!("dx: {dx:#?}");

    let h = 0.0003;

    fn err(claimed: f32, measured: f32) -> f32 {
        let d = measured.abs().max(0.01);
        (claimed - measured).abs() / d
    }

    let mut z_minus = Array::zeros(output_shape.clone());
    let mut z_plus = Array::zeros(output_shape.clone());
    for i in 0..n {
        // check accuracy of derivative at parameter params[i]
        let saved = params[i];
        params[i] = saved - h;
        layer.apply(params.view(), x.view(), tmp.view_mut(), z_minus.view_mut());
        params[i] = saved + h;
        layer.apply(params.view(), x.view(), tmp.view_mut(), z_plus.view_mut());
        params[i] = saved;

        let claimed = dp[i];
        let measured = ((&z_plus - &z_minus) * (1.0 / (2.0 * h)) * &dz).sum();

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
        layer.apply(params.view(), x.view(), tmp.view_mut(), z_minus.view_mut());
        x[i.clone()] = saved + h;
        layer.apply(params.view(), x.view(), tmp.view_mut(), z_plus.view_mut());
        x[i.clone()] = saved;

        let claimed = dx[i.clone()];
        let measured = ((&z_plus - &z_minus) * (1.0 / (2.0 * h)) * &dz).sum();

        let error = err(claimed, measured);
        assert!(
            error <= error_limit,
            "input element {i:?} computed derivative = {claimed}, measured = {measured}, error = {error}, limit = {error_limit}"
        );
    }
}

#[test]
fn test_layer_consistency() {
    test_layer(layers::LinearLayer::new(1, 1), (1, 1), 0.01);
    test_layer(layers::InputLayer::new().linear(1, 1), (1, 1), 0.01);
    test_layer(layers::LinearLayer::new(5, 3), (1, 5), 0.01);
    test_layer(layers::InputLayer::new().linear(5, 3), (1, 5), 0.01);

    test_layer(layers::LinearLayer::new(1, 1).relu(), (2, 1), 0.01);
    test_layer(layers::LinearLayer::new(10, 10).relu(), (2, 10), 0.01);
    test_layer(layers::LinearLayer::new(4, 2).softmax(), (3, 4), 0.01);
    test_layer(
        layers::LinearLayer::new(3, 3).relu().linear(3, 4).softmax(),
        (2, 3),
        0.01,
    );

    test_layer(layers::LinearLayer::new(1, 1).parallel(1), (1, 1), 0.01);
    test_layer(layers::LinearLayer::new(1, 1).parallel(1), (2, 1), 0.01);
    test_layer(layers::LinearLayer::new(1, 1).parallel(4), (4, 1), 0.01);
    test_layer(layers::LinearLayer::new(1, 1).parallel(4), (5, 1), 0.01);

    // test_layer(layers::LinearLayer::new(3, 2).relu().linear(2, 1).softmax(), (1, 3));
    // test_layer(layers::LinearLayer::new(30, 10).relu(), (2, 30));
    // test_layer(layers::LinearLayer::new(30, 10).relu().linear(10, 4), (2, 30));
    // test_layer(layers::LinearLayer::new(30, 10).relu().linear(10, 4).softmax(), (2, 30));

    test_layer(
        layers::Conv2dLayer::new(Ix4(2, 3, 3, 1)),
        (1, 4, 6, 1),
        0.04,
    );
    test_layer(
        layers::Conv2dLayer::new(Ix4(2, 3, 3, 3)),
        (2, 4, 6, 3),
        0.04,
    );
    // can't test MaxPool2dLayer in this way because `h` is too big
}

#[test]
fn test_pool() {
    let layer = layers::MaxPool2dLayer::new(2);
    let x: Array3<f32> = array![
        [[0.528783300, 0.48312938], [0.21382916, 0.98248150]],
        [[0.081051946, 0.60761390], [0.17663562, 0.98261246]]
    ];
    let x: Array4<f32> = x.into_shape((1, 2, 2, 2)).unwrap();

    let mut y: Array4<f32> = Array::zeros(layer.output_shape(x.raw_dim()));
    layer.apply(
        ArrayView::from_shape(0, &[]).unwrap(),
        x.view(),
        ArrayViewMut::from_shape((1, 0), &mut []).unwrap(),
        y.view_mut(),
    );
    assert_eq!(y.raw_dim().into_pattern(), (1, 1, 1, 2));
    assert_eq!(y.slice(s![0, 0, 0, ..]), array![0.5287833, 0.98261246]);

    let dy: Array4<f32> = 0.7f32 * Array::ones(layer.output_shape(x.raw_dim()));

    let dx = layer.derivatives(
        ArrayView::from_shape(0, &[]).unwrap(),
        x.view(),
        ArrayView2::from_shape((1, 0), &[]).unwrap(),
        dy.view(),
        ArrayViewMut1::from_shape(0, &mut []).unwrap(),
    );

    assert_eq!(
        dx.slice(s![0, .., .., ..]),
        array![[[0.7, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.7]],]
    );
}
