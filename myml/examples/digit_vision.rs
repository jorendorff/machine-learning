use std::time::Instant;

use mnist::*;
use ndarray::prelude::*;
use ndarray::Zip;

use myml::layers::InputLayer;
use myml::loss::CategoricalCrossEntropyLoss;
use myml::{Layer, Model};

fn preprocess(img: Vec<u8>, lbl: Vec<u8>) -> (Array4<f32>, Array1<usize>) {
    let n = lbl.len();
    let images = Array4::from_shape_vec((n, 28, 28, 1), img)
        .unwrap()
        .map(|x| *x as f32 / 255.0);
    let labels = Array1::from_shape_vec(n, lbl)
        .unwrap()
        .map(|x: &u8| *x as usize);
    (images, labels)
}

fn training_epochs<'a>(
    train_x: ArrayView4<'a, f32>,
    train_y: ArrayView1<'a, usize>,
    num_epochs: usize,
    batch_size: usize,
) -> impl Iterator<Item = impl IntoIterator<Item = (ArrayView4<'a, f32>, ArrayView1<'a, usize>)> + 'a> + 'a
{
    // TODO: shuffle images

    let n = train_x.len_of(Axis(0));
    (0..num_epochs).map(move |_epoch| {
        let mut train_x = train_x;
        let mut train_y = train_y;
        (0..n).step_by(batch_size).map(
            move |_begin| -> (ArrayView4<'a, f32>, ArrayView1<'a, usize>) {
                let remaining = train_x.len_of(Axis(0));
                let cut = batch_size.min(remaining);
                let (x, xs) = train_x.split_at(Axis(0), cut);
                let (y, ys) = train_y.split_at(Axis(0), cut);
                train_x = xs;
                train_y = ys;
                (x, y)
            },
        )
    })
}

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize();

    let (x_train, y_train) = preprocess(trn_img, trn_lbl);
    let (x_test, y_test) = preprocess(tst_img, tst_lbl);

    // let mut model = Model::new(
    //     FlattenLayer::new((1, 28, 28, 1).into_dimension())
    //         .linear(28 * 28, 500)
    //         .relu()
    //         .linear(500, 100)
    //         .relu()
    //         .linear(100, 10)
    //         .softmax()
    //         .parallel(8),
    //     CategoricalCrossEntropyLoss,
    // );
    // model.set_learning_rate(0.30); // with batch_size=256
    let mut model = Model::new(
        InputLayer::new()
            .conv_2d(32, 3, 3, 1)
            .relu()
            .max_pool_2d(2)
            .conv_2d(32, 3, 3, 32)
            .relu()
            .max_pool_2d(2)
            .flatten((1, 6, 6, 32))
            .linear(6 * 6 * 32, 128)
            .relu()
            .linear(128, 10)
            .softmax()
            .parallel(4),
        CategoricalCrossEntropyLoss,
    );
    model.set_learning_rate(0.25);

    let num_epochs = 5;
    let batch_size = 64;
    let t0 = Instant::now();
    model.train_epochs(training_epochs(
        x_train.view(),
        y_train.view(),
        num_epochs,
        batch_size,
    ));
    println!("trained {num_epochs} epochs in {:?}", t0.elapsed());

    let n_test = x_test.len_of(Axis(0));
    let yh = model.apply(x_test.view());

    let mut num_bad = 0;
    Zip::from(&y_test).and(yh.rows()).for_each(|&y, yh| {
        let p = yh[y];
        if yh.iter().any(|&x| x > p) {
            num_bad += 1;
        }
    });

    let accuracy = 100.0 * (n_test - num_bad) as f32 / n_test as f32;
    println!("{num_bad}/{n_test} test images misclassified ({accuracy:.1}% accuracy)");
}
