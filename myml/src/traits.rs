use std::fmt::Debug;

use ndarray::prelude::*;

use crate::layers::{ActivationLayer, BiasLayer, LinearLayer, Relu, Sequence, SoftmaxLayer};

pub trait Layer<D>: Debug
where
    D: Dimension,
{
    /// Type of the output shape, typically one of `Ix1`, `Ix2`, etc.
    ///
    /// Axis 0 of this is always the mini-batch axis.
    type Output: Dimension;

    /// Number of parameters required for this layer.
    ///
    /// The caller provides parameters to the other methods as a single flat
    /// array, which the methods will slice up and reshape into whatever they
    /// need.
    fn num_params(&self) -> usize {
        0
    }

    /// Return the output of this layer, given the `params` and the input `x`.
    ///
    /// Axis 0 of `x` is always the mini-batch axis; that is, each `x[i]` is a
    /// single training example or prediction task.
    fn apply(
        &self,
        params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
    ) -> Array<f32, Self::Output>;

    /// Given x and ∂L/∂z at x, compute partial derivatives ∂L/∂x and ∂L/∂p.
    ///
    /// Store ∂L/∂p in the out-param `dp`, a 1D vector of derivatives. Return
    /// ∂L/∂x.
    ///
    /// A step in backpropagation.
    ///
    /// `dz[i]` is the partial derivative of loss with respect to `z[i]`.
    /// It reflects the effect of that output as it propagates through the rest
    /// of the pipeline.
    fn derivatives(
        &self,
        params: ArrayView1<'_, f32>,
        x: ArrayView<'_, f32, D>,
        dz: ArrayView<'_, f32, Self::Output>,
        dp: ArrayViewMut1<'_, f32>,
    ) -> Array<f32, D>;

    fn then<L2>(self, other: L2) -> Sequence<Self, L2>
    where
        Self: Sized,
        L2: Layer<Self::Output>,
    {
        Sequence::new(self, other)
    }

    fn relu(self) -> Sequence<Self, ActivationLayer<Relu>>
    where
        Self: Sized,
    {
        self.then(ActivationLayer::new(Relu))
    }

    fn linear_no_bias(self, num_inputs: usize, num_outputs: usize) -> Sequence<Self, LinearLayer>
    where
        Self: Sized + Layer<D, Output = Ix2>,
    {
        Sequence::new(self, LinearLayer::new(num_inputs, num_outputs))
    }

    fn linear(
        self,
        num_inputs: usize,
        num_outputs: usize,
    ) -> Sequence<Self, Sequence<LinearLayer, BiasLayer<Ix2>>>
    where
        Self: Sized + Layer<D, Output = Ix2>,
    {
        let bias = BiasLayer::new(Ix2(1, num_outputs));

        self.then(Sequence::new(
            LinearLayer::new(num_inputs, num_outputs),
            bias,
        ))
    }

    fn softmax(self) -> Sequence<Self, SoftmaxLayer>
    where
        Self: Sized + Layer<D, Output = Ix2>,
    {
        Sequence::new(self, SoftmaxLayer)
    }
}

pub trait Loss<D: Dimension, Y>: Debug {
    fn loss(&self, y: Y, yh: ArrayView<'_, f32, D>) -> f32;
    fn accuracy(&self, y: Y, yh: ArrayView<'_, f32, D>) -> f32;
    fn deriv(&self, y: Y, yh: ArrayView<'_, f32, D>) -> Array<f32, D>;
}

pub trait ActivationFn: Copy + Clone + Debug {
    fn f(self, x: f32) -> f32;
    fn df(self, x: f32) -> f32;
}
