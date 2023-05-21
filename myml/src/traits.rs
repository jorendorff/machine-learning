use std::fmt::Debug;

use ndarray::prelude::*;

pub trait Layer<DI, DO>: Debug
where
    DI: Dimension,
    DO: Dimension,
{
    /// Number of parameters required for this layer.
    ///
    /// The caller provides parameters to the other methods as a single flat
    /// array, which the methods will slice up and reshape into whatever they
    /// need.
    fn num_params(&self) -> usize;

    /// Return the output of this layer, given the `params` and the input `x`.
    fn apply(&self, params: &Array<f32, Ix1>, x: &Array<f32, DI>) -> Array<f32, DO>;

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
        params: &Array<f32, Ix1>,
        x: &Array<f32, DI>,
        dz: &Array<f32, DO>,
        dp: &mut Array<f32, Ix1>,
    ) -> Array<f32, DI>;
}

pub trait Loss<D: Dimension, Y>: Debug {
    fn loss(&self, y: &Y, yh: &Array<f32, D>) -> f32;
    fn accuracy(&self, y: &Y, yh: &Array<f32, D>) -> f32;
    fn deriv(&self, y: &Y, yh: &Array<f32, D>) -> Array<f32, D>;
}
