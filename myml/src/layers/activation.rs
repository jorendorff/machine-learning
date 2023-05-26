use ndarray::prelude::*;
use ndarray::{RemoveAxis, Zip};

use crate::{ActivationFn, Layer};

/// Layer that applies the same real-valued function to each element.
#[derive(Debug)]
pub struct ActivationLayer<F> {
    f: F,
}

impl<F> ActivationLayer<F>
where
    F: ActivationFn,
{
    pub(crate) fn new(f: F) -> Self {
        ActivationLayer { f }
    }
}

impl<D, F> Layer<D> for ActivationLayer<F>
where
    D: Dimension + RemoveAxis,
    F: ActivationFn,
{
    type Output = D;

    fn output_shape(&self, input_shape: D) -> D {
        input_shape
    }

    fn apply(
        &self,
        _params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        _tmp: ArrayViewMut2<'_, f32>,
        mut y: ArrayViewMut<'_, f32, D>,
    ) {
        let f = self.f;
        y.assign(&x.mapv(move |v| f.f(v)));
    }

    fn derivatives(
        &self,
        _params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        _tmp: ArrayView2<'_, f32>,
        dz: ArrayView<'_, f32, D>,
        _dp: ArrayViewMut1<'_, f32>,
    ) -> Array<f32, D> {
        Zip::from(&x)
            .and(&dz)
            .map_collect(|&x, &dz| self.f.df(x) * dz)
    }
}

/// The logistic function, a handy symmetric, s-shaped function.
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl ActivationFn for Sigmoid {
    fn f(self, x: f32) -> f32 {
        sigmoid(x)
    }

    fn df(self, x: f32) -> f32 {
        let y = sigmoid(x);
        y * (1.0 - y)
    }
}

/// Rectified linear unit activation function.
#[derive(Debug, Clone, Copy)]
pub struct Relu;

impl ActivationFn for Relu {
    fn f(self, x: f32) -> f32 {
        if x >= 0.0 {
            x
        } else {
            0.0
        }
    }

    fn df(self, x: f32) -> f32 {
        if x >= 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
