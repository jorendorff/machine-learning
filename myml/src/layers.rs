use std::fmt::Debug;
use std::marker::PhantomData;

use ndarray::prelude::*;
use ndarray::{RemoveAxis, Zip};

use crate::{ActivationFn, Layer};

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
    D: Dimension,
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
        _tmp: ArrayViewMut1<'_, f32>,
        mut y: ArrayViewMut<'_, f32, D>,
    ) {
        let f = self.f;
        y.assign(&x.mapv(move |v| f.f(v)));
    }

    fn derivatives(
        &self,
        _params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        _tmp: ArrayView1<'_, f32>,
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

#[derive(Debug)]
pub struct Sequence<L1, L2> {
    first: L1,
    second: L2,
    first_num_params: usize,
    num_params: usize,
}

impl<L1, L2> Sequence<L1, L2> {
    pub(crate) fn new<D>(first: L1, second: L2) -> Self
    where
        D: Dimension,
        L1: Layer<D>,
        L2: Layer<L1::Output>,
    {
        // TODO: Assert first.output_shape() matches second.input_shape().
        let first_num_params = first.num_params();
        let num_params = first_num_params + second.num_params();
        Self {
            first,
            second,
            first_num_params,
            num_params,
        }
    }
}

impl<L1, L2, D> Layer<D> for Sequence<L1, L2>
where
    D: Dimension,
    L1: Layer<D>,
    L2: Layer<L1::Output>,
{
    type Output = L2::Output;

    fn output_shape(&self, input_shape: D) -> Self::Output {
        let hidden_shape = self.first.output_shape(input_shape);
        self.second.output_shape(hidden_shape)
    }

    fn num_params(&self) -> usize {
        self.num_params
    }

    fn num_hidden_activations(&self, input_shape: D) -> usize {
        let hidden_shape = self.first.output_shape(input_shape.clone());
        self.first.num_hidden_activations(input_shape)
            + hidden_shape.size()
            + self.second.num_hidden_activations(hidden_shape)
    }

    fn apply(
        &self,
        params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        tmp: ArrayViewMut1<'_, f32>,
        y: ArrayViewMut<'_, f32, Self::Output>,
    ) {
        let input_shape = x.raw_dim();
        let hidden_shape = self.first.output_shape(input_shape.clone());
        let (tmp1, tmp) = tmp.split_at(Axis(0), self.first.num_hidden_activations(input_shape));
        let (mid, tmp2) = tmp.split_at(Axis(0), hidden_shape.size());
        let mut m = mid
            .into_shape(hidden_shape)
            .expect("tmp should be contiguous and sized for first layer output");

        let (p1, p2) = params.split_at(Axis(0), self.first_num_params);
        self.first.apply(p1, x.view(), tmp1, m.view_mut());
        self.second.apply(p2, m.view(), tmp2, y);
    }

    fn derivatives(
        &self,
        params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        tmp: ArrayView1<'_, f32>,
        dz: ArrayView<'_, f32, Self::Output>,
        dp: ArrayViewMut1<'_, f32>,
    ) -> Array<f32, D> {
        let (p1, p2) = params.split_at(Axis(0), self.first_num_params);
        let (dp1, dp2) = dp.split_at(Axis(0), self.first_num_params);

        // Find the activations of the first layer. They are stored in the
        // middle of `tmp`. This is equivalent to recomputing them by calling
        // self.first.apply(), but that would be slow.
        let input_shape = x.raw_dim();
        let hidden_shape = self.first.output_shape(input_shape.clone());
        let m1 = self.first.num_hidden_activations(input_shape);
        let m2 = m1 + hidden_shape.size();
        let m = tmp
            .slice(s![m1..m2])
            .into_shape(hidden_shape)
            .expect("tmp should be contiguous and sized for first layer output");

        let dm = self.second.derivatives(p2, m, tmp.slice(s![m2..]), dz, dp2);
        self.first
            .derivatives(p1, x, tmp.slice(s![..m1]), dm.view(), dp1)
    }
}

#[derive(Debug)]
pub struct SoftmaxLayer;

impl Layer<Ix2> for SoftmaxLayer {
    type Output = Ix2;

    fn output_shape(&self, input_shape: Ix2) -> Ix2 {
        input_shape
    }

    fn apply(
        &self,
        _params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, Ix2>,
        _tmp: ArrayViewMut1<'_, f32>,
        mut y: ArrayViewMut2<'_, f32>,
    ) {
        let ex = x.mapv(|v| v.exp().clamp(1e-30, 1e30));
        let sum_ex = ex.sum_axis(Axis(1));
        y.assign(&(&ex / &sum_ex.slice(s![.., NewAxis])));
    }

    fn derivatives(
        &self,
        _params: ArrayView1<'_, f32>,
        x: ArrayView2<'_, f32>,
        _tmp: ArrayView1<'_, f32>,
        dz: ArrayView2<'_, f32>,
        _dp: ArrayViewMut1<'_, f32>,
    ) -> Array2<f32> {
        let ex = x.mapv(|v| v.exp().clamp(1e-15, 1e15));
        let sum_ex = ex.sum_axis(Axis(1));
        let sum_ex = sum_ex.slice(s![.., NewAxis]);

        // Cell (i,j) of the result should be
        //     sum(∂loss/∂z[i,k] * ∂z[i,k]/∂ex[i,j] * ∂ex[i,j]/∂x[i,j]
        //         for k in range(no))
        //   = sum(dz[i,k]
        //         * ((sum_ex[i,1] if j == k else 0) - ex[i,k]) / sum_ex[i,1]**2
        //         * ex[i,j]
        //         for k in range(no))
        //   = (ex[i,j] / sum_ex[i,1]**2)
        //     * sum(dz[i,k] * ((sum_ex[i,1] if i == k else 0) - ex[i,k])
        //           for k in range(no))
        //   = (ex[i,j] / sum_ex[i,1]**2)
        //     * (dz[i,j] * sum_ex[i,1]
        //        - sum(dz[i,k] * ex[i,k] for k in range(no)))

        (&ex / &sum_ex.mapv(|s| s.powi(2)))
            * (&dz * &sum_ex - (&dz * &ex).sum_axis(Axis(1)).slice(s![.., NewAxis]))
    }
}
