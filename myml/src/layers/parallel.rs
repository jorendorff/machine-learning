use ndarray::prelude::*;
use ndarray::RemoveAxis;
use rayon::prelude::*;

use crate::Layer;

#[derive(Debug)]
pub struct ParallelLayer<L> {
    /// Layer being parallelized (typically the whole pipeline).
    inner: L,
    /// Number of training samples to put in each Rayon task.
    batch_size: usize,
}

impl<L> ParallelLayer<L> {
    pub fn new(inner: L, batch_size: usize) -> Self {
        assert!(batch_size > 0);
        ParallelLayer { inner, batch_size }
    }
}

impl<D, L> Layer<D> for ParallelLayer<L>
where
    D: Dimension + RemoveAxis,
    L: Layer<D> + Sync,
{
    type Output = L::Output;

    fn output_shape(&self, input_shape: D) -> Self::Output {
        self.inner.output_shape(input_shape)
    }

    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    fn hidden_activations_shape(&self, input_shape: D) -> Ix2 {
        self.inner.hidden_activations_shape(input_shape)
    }

    fn apply(
        &self,
        params: ndarray::ArrayView1<'_, f32>,
        x: ndarray::ArrayView<'_, f32, D>,
        mut tmp: ndarray::ArrayViewMut2<'_, f32>,
        mut y: ndarray::ArrayViewMut<f32, Self::Output>,
    ) {
        let xi = x.axis_chunks_iter(Axis(0), self.batch_size).into_par_iter();
        let tmpi = tmp
            .axis_chunks_iter_mut(Axis(0), self.batch_size)
            .into_par_iter();
        let yi = y
            .axis_chunks_iter_mut(Axis(0), self.batch_size)
            .into_par_iter();

        xi.zip(yi).zip(tmpi).for_each(|((x, y), tmp)| {
            self.inner.apply(params, x, tmp, y);
        });
    }

    fn derivatives(
        &self,
        params: ndarray::ArrayView1<'_, f32>,
        x: ndarray::ArrayView<'_, f32, D>,
        tmp: ndarray::ArrayView2<'_, f32>,
        dz: ndarray::ArrayView<'_, f32, Self::Output>,
        mut dp: ndarray::ArrayViewMut1<'_, f32>,
    ) -> ndarray::Array<f32, D> {
        let num_samples = x.len_of(Axis(0));
        let num_batches = (num_samples + self.batch_size - 1) / self.batch_size;

        let xi = x.axis_chunks_iter(Axis(0), self.batch_size).into_par_iter();
        let tmpi = tmp
            .axis_chunks_iter(Axis(0), self.batch_size)
            .into_par_iter();
        let dzi = dz
            .axis_chunks_iter(Axis(0), self.batch_size)
            .into_par_iter();
        let mut dpw = Array2::zeros((num_batches, dp.len()));
        let dpwi = dpw.axis_iter_mut(Axis(0)).into_par_iter();
        let mut dx = Array::zeros(x.raw_dim());
        let dxi = dx
            .axis_chunks_iter_mut(Axis(0), self.batch_size)
            .into_par_iter();

        xi.zip(tmpi)
            .zip(dzi)
            .zip(dpwi)
            .zip(dxi)
            .for_each(|((((x, tmp), dz), mut dpw), mut dx)| {
                dx.assign(&self.inner.derivatives(params, x, tmp, dz, dpw.view_mut()));
            });

        dp.assign(&dpw.sum_axis(Axis(0)));
        dx
    }
}
