//! Max pooling.

use ndarray::prelude::*;

use crate::Layer;

/// Max pooling operation for image data.
///
/// The input shape is `(num_images, height, width, num_channels)`.
#[derive(Debug)]
pub struct MaxPool2dLayer {
    size: usize,
}

impl MaxPool2dLayer {
    pub fn new(size:usize)->Self{MaxPool2dLayer {size}}
}

impl Layer<Ix4> for MaxPool2dLayer {
    type Output = Ix4;

    fn output_shape(&self, input_shape: Ix4) -> Ix4 {
        let (xn, xh, xw, xc) = input_shape.into_pattern();
        Ix4(xn, (xh + self.size - 1) / self.size, (xw + self.size - 1) / self.size, xc)
    }

    fn apply(
        &self,
        _params: ArrayView1<f32>,
        x: ArrayView4<f32>,
        _tmp: ArrayViewMut2<f32>,
        mut y: ArrayViewMut4<f32>,
    ) {
        // minibatch axis
        for (x, mut y) in x.outer_iter().zip(y.outer_iter_mut()) {
            // y axis
            for (x, mut y) in x.axis_chunks_iter(Axis(0), self.size).zip(y.outer_iter_mut()) {
                // x axis
                for (x, mut y) in x.axis_chunks_iter(Axis(1), self.size).zip(y.outer_iter_mut()) {
                    debug_assert_eq!(y.len(), x.len_of(Axis(2)));
                    // channel axis
                    for (x, y) in x.axis_iter(Axis(2)).zip(y.iter_mut()) {
                        // now x[0..n,0..n] is a little square (or rectangle) of pixels and y is
                        // the output cell.
                        *y = x.iter().copied().fold(f32::MIN, f32::max);
                    }
                }
            }
        }
    }

    fn derivatives(
        &self,
        _params: ArrayView1<f32>,
        x: ArrayView4<f32>,
        _tmp: ArrayView2<f32>,
        dz: ArrayView4<f32>,
        _dp: ArrayViewMut1<f32>,
    ) -> Array4<f32> {
        // redo the work :-\
        let mut z = Array4::zeros(dz.raw_dim());
        self.apply(
            ArrayView::from_shape(0, &[]).unwrap(),
            x,
            ArrayViewMut::from_shape((x.len_of(Axis(0)), 0), &mut[]).unwrap(),
            z.view_mut(),
        );

        // ∂z/∂x is 1 exactly where an element's value is equal to the max,
        // and 0 elsewhere.
        let mut dx = Array::zeros(x.raw_dim());
        let n = self.size;

        // minibatch axis
        for ((x, mut dx), (z, dz)) in x.outer_iter().zip(dx.outer_iter_mut()).zip(z.outer_iter().zip(dz.outer_iter())) {
            // y axis
            for ((x, mut dx), (z, dz)) in x.axis_chunks_iter(Axis(0), n).zip(dx.axis_chunks_iter_mut(Axis(0), n)).zip(z.outer_iter().zip(dz.outer_iter())) {
                // x axis
                for ((x, mut dx), (z, dz)) in x.axis_chunks_iter(Axis(1), n).zip(dx.axis_chunks_iter_mut(Axis(1), n)).zip(z.outer_iter().zip(dz.outer_iter())) {
                    // channel axis
                    for ((x, mut dx), (z, dz)) in x.axis_iter(Axis(2)).zip(dx.axis_iter_mut(Axis(2))).zip(z.iter().zip(dz.iter())) {
                        // x and dx are little squares (or rectangles); z and dz are floats.
                        let (h, w) = x.raw_dim().into_pattern();
                        for i in 0..h {
                            for j in 0..w {
                                dx[[i, j]] = if x[[i, j]] == *z {  *dz } else { 0.0 };
                            }
                        }
                    }
                }
            }
        }
        dx
    }
}
