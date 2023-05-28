use ndarray::prelude::*;
use ndarray::RemoveAxis;

use crate::Layer;

/// Flatten a shape from n-dimensional to 2-dimensional.
///
/// Axis 0, the mini-batch axis, is retained.
pub(crate) fn flatten<D: RemoveAxis>(shape: D) -> Ix2 {
    Ix2(shape[0], shape.remove_axis(Axis(0)).size())
}

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
    D: Dimension + RemoveAxis,
{
    pub fn new(mut input_shape: D) -> Self {
        input_shape.as_array_view_mut()[0] = 1;
        FlattenLayer { input_shape }
    }
}

impl<D: Dimension + RemoveAxis> Layer<D> for FlattenLayer<D> {
    type Output = Ix2;

    fn output_shape(&self, input_shape: D) -> Ix2 {
        flatten(input_shape)
    }

    fn apply(
        &self,
        _params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        _tmp: ArrayViewMut2<'_, f32>,
        mut y: ArrayViewMut2<'_, f32>,
    ) {
        assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().copied().zip(y.iter_mut()) {
            *y = x;
        }
    }

    fn derivatives(
        &self,
        _params: ArrayView1<'_, f32>,
        _x: ArrayView<'_, f32, D>,
        _tmp: ArrayView2<'_, f32>,
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
