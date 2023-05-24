use ndarray::prelude::*;

use crate::Layer;

/// Layer that converts each vector of the input to a probability distribution,
/// using the [softmax function](https://en.wikipedia.org/wiki/Softmax_function).
///
/// This is often the last layer in a classification network.
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
