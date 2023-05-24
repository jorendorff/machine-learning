use std::marker::PhantomData;

use ndarray::prelude::*;

use crate::Layer;

/// Layer that does nothing but specify the shape of the data coming in.
#[derive(Debug)]
pub struct InputLayer<D> {
    phantom: PhantomData<D>,
}

impl<D: Dimension> InputLayer<D> {
    #[allow(dead_code)]
    pub fn new() -> Self {
        InputLayer {
            phantom: PhantomData,
        }
    }
}

impl<D: Dimension> Layer<D> for InputLayer<D> {
    type Output = D;

    fn output_shape(&self, input_shape: D) -> D {
        input_shape
    }

    fn apply(
        &self,
        _params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        _tmp: ArrayViewMut1<'_, f32>,
        mut y: ArrayViewMut<'_, f32, D>,
    ) {
        y.assign(&x);
    }

    fn derivatives(
        &self,
        _params: ArrayView1<'_, f32>,
        _x: ArrayView<'_, f32, D>,
        _tmp: ArrayView1<'_, f32>,
        dz: ArrayView<'_, f32, D>,
        _dp: ArrayViewMut1<'_, f32>,
    ) -> Array<f32, D> {
        dz.into_owned()
    }
}
