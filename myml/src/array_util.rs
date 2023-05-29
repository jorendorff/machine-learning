use ndarray::prelude::*;

fn compute_strides_splitting<D1, D2>(
    view_shape: D1,
    view_strides: &[isize],
    new_shape: D2,
) -> D2
where
    D1: Dimension,
    D2: Dimension,
{
    assert_eq!(
        view_shape.size(),
        new_shape.size(),
        "existing view with shape {view_shape:?} is incompatible with new shape {new_shape:?}",
    );

    let new_dims = new_shape.as_array_view();
    let mut new_strides = D2::zeros(new_dims.len());
    let mut new_strides_view = new_strides.as_array_view_mut();
    let mut i = 0;
    let mut new_dims = new_dims.iter().copied();

    let view_dims = view_shape.as_array_view();
    for (&stride, &len) in view_strides.iter().zip(view_dims) {
        let mut remaining = len;
        while remaining != 1 {
            let d = new_dims
                .next()
                .expect("already checked that product matches shape");
            assert_eq!(
                remaining % d,
                0,
                "existing view with shape {view_shape:?} is incompatible with new shape {new_shape:?}",
            );
            remaining /= d;
            let ir = remaining as isize;
            assert!(ir > 0, "array size should be reasonable");
            new_strides_view[i] =
                stride
                    .checked_mul(ir)
                    .expect("array arithmetic can't overflow if existing view was ok")
                    as usize;
            i += 1;
        }
    }
    for spare_dim in new_dims {
        assert_eq!(spare_dim, 1);
        new_strides_view[i] = 0;
        i += 1;
    }
    assert_eq!(i, new_strides_view.len());

    new_strides
}

/// Reshape the given view.
///
/// The new shape must be the same size and must have the effect of splitting
/// existing axes without reordering them. Some examples of reshapes that do
/// and don't work:
///
/// -   `100` to `(10, 10)` - ok
/// -   `(10, 10)` to `100` - bad
/// -   `100` to `(2, 2, 5, 5)` - ok
/// -   `(2, 2, 5, 5)` to `100` - bad
/// -   `(6, 5)` to `(2, 3, 5)` - ok
/// -   `(6, 5)` to `(3, 5, 2)` - bad
///
/// This works by recomputing the strides. The strides of axes that aren't split
/// are unaffected.
///
/// The existing elements appear distributed along the new axes in "C" order.
///
/// *Panics* if the new shape is bad.
#[allow(dead_code)]
pub fn reshape_splitting<'a, T, D1, D2>(
    view: ArrayView<'a, T, D1>,
    shape: D2,
) -> ArrayView<'a, T, D2>
where
    D1: Dimension,
    D2: Dimension,
{
    let view_shape = view.raw_dim();
    let new_shape = shape.clone().into_shape().raw_dim().clone();
    let strides = compute_strides_splitting(view_shape, view.strides(), new_shape);

    unsafe { ArrayView::from_shape_ptr(shape.strides(strides), view.as_ptr()) }
}

pub fn reshape_splitting_mut<'a, T, D1, D2>(
    view: ArrayViewMut<'a, T, D1>,
    shape: D2,
) -> ArrayViewMut<'a, T, D2>
where
    D1: Dimension,
    D2: Dimension,
{
    let view_shape = view.raw_dim();
    let new_shape = shape.clone().into_shape().raw_dim().clone();
    let strides = compute_strides_splitting(view_shape, view.strides(), new_shape);

    let ptr = view.as_ptr() as *mut T;
    drop(view);
    unsafe { ArrayViewMut::from_shape_ptr(shape.strides(strides), ptr) }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn striding() {
        let a = (0..200).collect::<Array1<i32>>();
        let v = a.slice(s![30..130]);
        let v2 = reshape_splitting(v, Ix2(10, 10));
        assert_eq!(v2[[0, 0]], 30);
        assert_eq!(v2[[0, 1]], 31);
        assert_eq!(v2[[1, 0]], 40);

        let v3 = reshape_splitting(v, Ix4(2, 2, 5, 5));
        assert_eq!(v3[[0, 0, 0, 0]], 30);
        assert_eq!(v3[[0, 0, 0, 1]], 31);
        assert_eq!(v3[[0, 0, 1, 0]], 35);
        assert_eq!(v3[[0, 1, 0, 0]], 55);
        assert_eq!(v3[[1, 0, 0, 0]], 80);

        let v = a.slice(s![100..130]);
        let v4 = reshape_splitting(v, Ix2(6, 5));
        let v5 = reshape_splitting(v4, Ix3(2, 3, 5));
        assert_eq!(v5[[0, 0, 0]], 100);
        assert_eq!(v5[[0, 0, 1]], 101);
        assert_eq!(v5[[0, 1, 0]], 105);
        assert_eq!(v5[[1, 0, 0]], 115);
    }

    #[test]
    #[should_panic]
    fn bad_split_1() {
        let a: Array2<f32> = Array::zeros((10, 10));
        reshape_splitting(a.view(), Ix1(100));
    }

    #[test]
    #[should_panic]
    fn bad_split_2() {
        let a: Array4<f32> = Array::zeros((2, 2, 5, 5));
        reshape_splitting(a.view(), Ix3(2, 2, 25));
    }

    #[test]
    #[should_panic]
    fn bad_split_3() {
        let a: Array2<f32> = Array::zeros((6, 5));
        reshape_splitting(a.view(), Ix3(3, 5, 2));
    }
}
