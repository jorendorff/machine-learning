use ndarray::prelude::*;
use ndarray::RemoveAxis;

use crate::Layer;

/// A dense layer with a matrix of weights. Note: no biases! Use a BiasLayer
/// immediately after.
#[derive(Debug)]
pub struct LinearLayer {
    /// Number of inputs. Input shape is `(N, ni)`.
    ni: usize,
    /// Number of cells, i.e. outputs. Output shape is `(N, no)`.
    no: usize,
}

impl LinearLayer {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        LinearLayer {
            ni: num_inputs,
            no: num_outputs,
        }
    }
}

impl Layer<Ix2> for LinearLayer {
    type Output = Ix2;

    fn output_shape(&self, input_shape: Ix2) -> Ix2 {
        Ix2(input_shape[0], self.no)
    }

    fn num_params(&self) -> usize {
        self.ni * self.no
    }

    fn apply(
        &self,
        params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, Ix2>,
        _tmp: ArrayViewMut1<'_, f32>,
        mut y: ArrayViewMut2<'_, f32>,
    ) {
        let ni = self.ni;
        let no = self.no;
        assert_eq!(x.shape()[1], ni);
        let w = params
            .into_shape((ni, no))
            .expect("size of params should be self.num_params()");
        y.assign(&x.dot(&w))
    }

    fn derivatives(
        &self,
        params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, Ix2>,
        _tmp: ArrayView1<'_, f32>,
        dz: ArrayView<'_, f32, Self::Output>,
        mut dp: ArrayViewMut1<'_, f32>,
    ) -> Array<f32, Ix2> {
        let ni = self.ni;
        let no = self.no;
        let n = x.shape()[0];
        assert_eq!(x.shape()[1], ni);

        let w = params
            .into_shape((ni, no))
            .expect("size of params should be self.num_params()");
        assert_eq!(dz.shape(), [n, no]);

        let dw = x.t().dot(&dz);
        assert_eq!(dw.shape(), [ni, no]);
        dp.assign(
            &dw.into_shape(ni * no)
                .expect("just asserted this is the right size"),
        );

        let dx = dz.dot(&w.t());
        assert_eq!(dx.shape(), x.shape());
        dx
    }
}

/// Layer that adds a parameter to each input. The output shape is the same as
/// the input shape. The first dimension is assumed to be the example axis, so
/// if the input shape is `(N, k)` then there are _k_ parameters.
#[derive(Debug)]
pub struct BiasLayer<D> {
    shape: D,
}

impl<D: Dimension> BiasLayer<D> {
    pub(crate) fn new(mut shape: D) -> Self {
        shape.as_array_view_mut()[0] = 1;
        BiasLayer { shape }
    }
}

impl<D> Layer<D> for BiasLayer<D>
where
    D: Dimension + RemoveAxis,
{
    type Output = D;

    fn output_shape(&self, input_shape: D) -> D {
        input_shape
    }

    fn num_params(&self) -> usize {
        self.shape
            .size_checked()
            .expect("overflow in size calculation")
    }

    fn apply(
        &self,
        params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        _tmp: ArrayViewMut1<'_, f32>,
        mut y: ArrayViewMut<'_, f32, D>,
    ) {
        y.assign(
            &(&x + &params
                .into_shape(self.shape.clone())
                .expect("params size should match self.num_params()")),
        );
    }

    fn derivatives(
        &self,
        _params: ArrayView1<'_, f32>,
        _x: ArrayView<'_, f32, D>,
        _tmp: ArrayView1<'_, f32>,
        dz: ArrayView<'_, f32, D>,
        mut dp: ArrayViewMut1<'_, f32>,
    ) -> Array<f32, D> {
        let dp_shape = dp.raw_dim();
        dp.assign(&dz.sum_axis(Axis(0)).into_shape(dp_shape).expect(
            "number of parameters should be the same as the number of outputs per example",
        ));
        dz.into_owned()
    }
}
