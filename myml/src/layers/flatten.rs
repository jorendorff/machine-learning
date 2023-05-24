use ndarray::prelude::*;

use crate::Layer;

/// Reshape inputs to matrix form.
///
/// This layer takes inputs with example-index as the first axis and any number
/// of other axes. It flattens each example into a row, leaving a 2D matrix.
///
/// That is, the actual input shape is `input_shape`, except with the first
/// dimension changed to the number of examples; and the output shape is
/// `(num_examples, input_shape.size())`.
#[derive(Debug)]
pub struct FlattenLayer<D> {
    /// The input shape of this layer, for a batch with one example.
    ///
    /// Invariant: `input_shape.as_array_view()[0] == 1`
    input_shape: D,
}

impl<D> FlattenLayer<D>
where
    D: Dimension,
{
    pub fn new(mut input_shape: D) -> Self {
        input_shape.as_array_view_mut()[0] = 1;
        FlattenLayer { input_shape }
    }
}

impl<D: Dimension> Layer<D> for FlattenLayer<D> {
    type Output = Ix2;

    fn output_shape(&self, input_shape: D) -> Ix2 {
        let n = input_shape[0];
        Ix2(n, input_shape.size() / n)
    }

    fn apply(
        &self,
        _params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        _tmp: ArrayViewMut1<'_, f32>,
        mut y: ArrayViewMut2<'_, f32>,
    ) {
        let size = self.input_shape.size();
        let n = x.shape()[0];
        y.assign(
            &x.into_shape((n, size))
                .expect("rows of x should match self.input_shape"),
        );
    }

    fn derivatives(
        &self,
        _params: ArrayView1<'_, f32>,
        _x: ArrayView<'_, f32, D>,
        _tmp: ArrayView1<'_, f32>,
        dz: ArrayView<'_, f32, Ix2>,
        _dp: ArrayViewMut1<'_, f32>,
    ) -> Array<f32, D> {
        let mut shape = self.input_shape.clone();
        shape.as_array_view_mut()[0] = dz.shape()[0];
        dz.into_shape(shape)
            .expect("rows of dz should match self.input_shape")
            .into_owned()
    }
}
