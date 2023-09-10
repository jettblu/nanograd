use crate::{ Tensor, TensorTrait, Dimensions, DataArray, types::ops::UnaryOps, Ops };

/// Raise each value in tensor to power of val
///
/// # Arguments
///
/// * `val` - The value to raise each value in tensor to.
fn exp<T: TensorTrait<T>>(base: T, power: Tensor<T>) -> Tensor<T> {
    // throw not implemented for now, since backward op is not defined
    panic!("Not implemented");
    let dim: Dimensions = power.dim();
    let mut i: usize = 0;
    let mut new_data = Vec::with_capacity(dim.0 * dim.1);
    let data: &DataArray<T> = power.data();
    while i < dim.0 * dim.1 {
        new_data.push(data[i].pow(base));
        i += 1;
    }
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    // create and return a new tensor
    let mut new_tensor = Tensor::_build_raw(
        new_data,
        dim,
        None,
        Some(true),
        Some(Ops::UnaryOps(UnaryOps::EXP2)),
        Some(vec![power])
    );
    new_tensor.set_gradient(Tensor::zeros(dim, None, None));
    new_tensor
}

/// Compute 2 raised to the power of each value in tensor.
///
/// # Arguments
///
/// * `base` - The tensor to raise each value in tensor to power of 2
///
/// # Returns
///
/// A tensor where each value is 2 raised to the power of the corresponding value in tensor.
///
/// # Examples
///
/// ```
/// use nanograd::{ Tensor, exp2 };
///
/// let tensor:Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice(), (2, 2), None, None);
/// let tensor_exp2 = exp2(tensor);
/// ```
///
pub fn exp2<T: TensorTrait<T>>(power: Tensor<T>) -> Tensor<T> {
    exp(T::from_f32(2.0).unwrap(), power)
}
