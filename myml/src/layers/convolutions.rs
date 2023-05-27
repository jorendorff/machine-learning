//! Convolutions for image processing.
//!
//! Convolution layers take input of the shape `(num_images, img_height, img_width, num_img_channels)`.
//!
//! Convolution kernels have the shape `(num_output_channels, ker_height, ker_width, num_image_channels)`
//!
//! Kernel sizes are `(height, width)`.

use ndarray::prelude::*;

use crate::Layer;

/// Convolve matrices in 2D as needed for a convolutional neural network.
///
/// The shape of `images` is `(num_images, img_height, img_width, num_img_channels)`.
/// The shape of `kernel` is `(num_output_channels, ker_height, ker_width, num_img_channels)`.
/// Note that `num_img_channels` must agree.
/// The output shape is `(new_img_width, new_img_height, num_output_channels, num_images)`.
///
/// Implementation is after <https://stackoverflow.com/a/64660822>.
fn conv2d_impl(images: ArrayView4<f32>, kernel: ArrayView4<f32>, mut z: ArrayViewMut4<f32>) {
    let (xn, xh, xw, xc) = images.raw_dim().into_pattern();
    let (kn, kh, kw, kc) = kernel.raw_dim().into_pattern();
    assert_eq!(kc, xc, "incompatible number of channels: images={xc}, kernel={kc}");
    let ow = xw - kw + 1;
    let oh = xh - kh + 1;
    assert_eq!(z.raw_dim().into_pattern(), (xn, oh, ow, kn));
    for t in 0..xn {
        for y in 0..oh {
            for x in 0..ow {
                for oc in 0..kn {
                    for j in 0..kh {
                        for i in 0..kw {
                            for ic in 0..xc {
                                z[[t, y, x, oc]] += kernel[[oc, j, i, ic]] * images[[t, y + j, x + i, ic]];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Compute derivatives of loss with respect to Conv2D kernel parameters.
fn conv2d_dk(images: ArrayView4<f32>, dz: ArrayView4<f32>, mut dk: ArrayViewMut4<f32>) {
    let (xn, xh, xw, xc) = images.raw_dim().into_pattern();
    let (zn, zh, zw, zc) = dz.raw_dim().into_pattern();
    assert_eq!(zn, xn, "incompatible number of samples: images={xn}, dz={zn}");
    let kw = xw - zw + 1;
    let kh = xh - zh + 1;
    assert_eq!(dk.raw_dim().into_pattern(), (zc, kh, kw, xc));
    for t in 0..xn {
        for oy in 0..zh {
            for ox in 0..zw {
                for co in 0..zc {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            for ci in 0..xc {
                                dk[[co, ky, kx, ci]] += images[[t, oy + ky, ox + kx, ci]] * dz[[t, oy, ox, co]];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Compute derivatives of loss with respect to Conv2D input image pixels.
fn conv2d_dx(kernel: ArrayView4<f32>, dz: ArrayView4<f32>, mut dx: ArrayViewMut4<f32>) {
    let (kc, kh, kw, xc) = kernel.raw_dim().into_pattern();
    let (n, zh, zw, zc) = dz.raw_dim().into_pattern();
    assert_eq!(kc, zc, "incompatible number of channels: kernel={kc}, dz={zc}");
    let xh = zh + kh - 1;
    let xw = zw + kw - 1;
    assert_eq!(dx.raw_dim().into_pattern(), (n, xh, xw, xc));
    for t in 0..n {
        for zy in 0..zh {
            for zx in 0..zw {
                for co in 0..zc {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            for ci in 0..xc {
                                dx[[t, zy + ky, zx + kx, ci]] += kernel[[co, ky, kx, ci]] * dz[[t, zy, zx, co]];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 2D convolution with no padding.
///
/// In addition to the convolution kernel, each output channel gets a bias, a
/// constant added to each pixel of that channel.
#[derive(Debug)]
pub struct Conv2dLayer {
    kernel_shape: Ix4,
}

impl Conv2dLayer {
    /// Create a convolutional layer.
    ///
    /// kernel_shape must be (num_output_channels, height, width, num_img_channels).
    ///
    /// For example, to use 64 3x3 filters on a grayscale image,
    /// use `(64, 3, 3, 1)`.
    pub fn new(kernel_shape: Ix4) -> Self {
        Conv2dLayer { kernel_shape }
    }
}

impl Layer<Ix4> for Conv2dLayer {
    type Output = Ix4;

    fn output_shape(&self, input_shape: Ix4) -> Self::Output {
        let (xn, xh, xw, xc) = input_shape.into_pattern();
        let (oc, kh, kw, ic) = self.kernel_shape.into_pattern();
        assert_eq!(ic, xc, "incompatible number of channels: images={xc}, kernel={ic}");
        Ix4(xn, xh - kh + 1, xw - kw + 1, oc)
    }

    fn num_params(&self) -> usize {
        let (oc, kh, kw, ic) = self.kernel_shape.into_pattern();
        oc * kh * kw * ic + oc
    }

    fn apply(
        &self,
        params: ArrayView1<f32>,
        x: ArrayView4<f32>,
        _tmp: ArrayViewMut2<f32>,
        mut y: ArrayViewMut4<f32>,
    ) {
        let ks = self.kernel_shape.size();
        let kernel = params.slice(s![..ks]).into_shape(self.kernel_shape).expect("params must be contiguous");
        let bias = params.slice(s![ks..]);
        y.assign(&bias);
        conv2d_impl(x, kernel, y);
    }

    fn derivatives(
        &self,
        params: ArrayView1<f32>,
        x: ArrayView4<f32>,
        _tmp: ArrayView2<f32>,
        dz: ArrayView4<f32>,
        mut dp: ArrayViewMut1<f32>,
    ) -> Array4<f32> {
        let ks = self.kernel_shape.size();
        dp.iter_mut().for_each(|dp| *dp = 0.0);
        conv2d_dk(x, dz, dp.slice_mut(s![..ks]).into_shape(self.kernel_shape).expect("dp must be contiguous"));
        dp.slice_mut(s![ks..]).assign(&dz.sum_axis(Axis(2)).sum_axis(Axis(1)).sum_axis(Axis(0)));
        let kernel = params.slice(s![..ks]).into_shape(self.kernel_shape).expect("params must be contiguous");
        let mut dx = Array::zeros(x.raw_dim());
        conv2d_dx(kernel, dz, dx.view_mut());
        dx
    }
}
