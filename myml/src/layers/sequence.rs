use ndarray::prelude::*;

use crate::Layer;

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
