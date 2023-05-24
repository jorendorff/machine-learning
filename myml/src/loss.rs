//! Loss functions.

use ndarray::prelude::*;
use ndarray::Zip;

use crate::traits::Loss;

/// Loss function for logistic regression. AKA binary cross-entropy.
#[derive(Debug, Clone, Copy)]
pub struct LogisticLoss;

impl<'a> Loss<Ix1, ArrayView1<'a, bool>> for LogisticLoss {
    fn loss(&self, y: ArrayView1<'a, bool>, yh: ArrayView1<'_, f32>) -> f32 {
        Zip::from(&y)
            .and(yh)
            .map_collect(|&y, &yh| {
                let err = if y { yh } else { 1.0 - yh };
                -err.ln()
            })
            .mean()
            .unwrap_or(0.0)
    }

    // Compute partial derivative of loss with respect to yh.
    fn deriv(&self, y: ArrayView1<'a, bool>, yh: ArrayView1<'_, f32>) -> Array1<f32> {
        // When y == 0, we want the derivative of -log(1-yh)/n, or 1/(n*(1-yh)).
        // When y == 1, we want the derivative of -log(yh)/n, or -1/(n*yh).
        assert_eq!(y.shape(), yh.shape());
        let n = y.shape()[0];
        Zip::from(&y).and(yh).map_collect(|&y, &yh| {
            let d = if y { -yh } else { 1.0 - yh };
            1.0 / (n as f32 * d)
        })
    }

    fn accuracy(&self, y: ArrayView1<'a, bool>, yh: ArrayView1<'_, f32>) -> f32 {
        let n = y.shape()[0];
        let mut num_good = 0;
        Zip::from(y).and(yh).for_each(|&y, &yh| {
            if y == (yh >= 0.5) {
                num_good += 1;
            }
        });
        num_good as f32 / n as f32
    }
}

/// A loss function for classification problems.
///
/// The output `yh` of the network must be normalized e.g. by using a
/// `SoftmaxLayer`.
#[derive(Debug, Clone, Copy)]
pub struct CategoricalCrossEntropyLoss;

impl<'a> Loss<Ix2, ArrayView1<'a, usize>> for CategoricalCrossEntropyLoss {
    fn loss(&self, y: ArrayView1<'a, usize>, yh: ArrayView2<'_, f32>) -> f32 {
        let n = yh.shape()[0]; // number of examples
        let c = yh.shape()[1]; // number of categories
        assert_eq!(y.shape(), &[n]);
        assert!(y.iter().all(|&y| y < c));
        assert!(yh.iter().all(|&yh| 0.0 <= yh && yh <= 1.0));
        if n == 0 {
            0.0
        } else {
            let mut total = 0.0;
            Zip::from(y).and(yh.rows()).for_each(|&y, yh| {
                let p = yh[y].max(f32::MIN_POSITIVE);
                assert!(0.0 < p && p <= 1.0);
                total += -p.ln();
            });
            total / n as f32
        }
    }

    fn deriv(&self, y: ArrayView1<'a, usize>, yh: ArrayView2<'_, f32>) -> Array2<f32> {
        let n = yh.shape()[0]; // number of examples
        let c = yh.shape()[1]; // number of categories
        assert_eq!(y.shape(), &[n]);
        let mut dyh = Array2::<f32>::zeros((n, c));

        Zip::from(dyh.rows_mut())
            .and(y)
            .and(yh.rows())
            .for_each(|mut dyh, &y, yh| dyh[y] = -1.0 / (n as f32 * yh[y]));
        dyh
    }

    fn accuracy(&self, y: ArrayView1<'a, usize>, yh: ArrayView2<'_, f32>) -> f32 {
        let n = yh.shape()[0]; // number of examples
        assert_eq!(y.shape(), &[n]);
        let mut num_good = 0;
        Zip::from(y).and(yh.rows()).for_each(|&y, yh| {
            let p = yh[y];
            if yh.iter().all(|&x| x <= p) {
                num_good += 1;
            }
        });
        num_good as f32 / n as f32
    }
}
