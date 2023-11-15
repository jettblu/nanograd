use std::f64::consts::E;

use crate::{ Tensor, TensorTrait, Dimensions, DataArray, types::ops::{ UnaryOps, ReduceOps }, Ops };

/// Raise each value in tensor to power of val
///
/// # Arguments
///
/// * `val` - The value to raise each value in tensor to.
fn exp<T: TensorTrait<T>>(base: T, power: Tensor<T>) -> Tensor<T> {
    // throw not implemented for now, since backward op is not defined
    if base != T::from_f32(2.0).unwrap() {
        panic!("Not implemented for bases other than 2.0");
    }
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
        Some(power),
        None
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
/// use nanograd::{ Tensor, nn::transformation::exp2 };
///
/// let tensor:Tensor<f64> = Tensor::new(vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice(), (2, 2), None, None);
/// let tensor_exp2 = exp2(tensor);
/// ```
///
pub fn exp2<T: TensorTrait<T>>(power: Tensor<T>) -> Tensor<T> {
    exp(T::from_f32(2.0).unwrap(), power)
}

pub fn max<T: TensorTrait<T>>(val: Tensor<T>, other: T) -> Tensor<T> {
    let dim: Dimensions = val.dim();
    let mut new_data = Vec::with_capacity(dim.0 * dim.1);
    let data: &DataArray<T> = val.data();
    for i in 0..dim.0 * dim.1 {
        new_data.push(if data[i] > other { data[i] } else { other });
    }
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    let mut new_tensor = Tensor::_build_raw(
        new_data,
        dim,
        None,
        Some(true),
        Some(Ops::UnaryOps(UnaryOps::MAX)),
        Some(val),
        None
    );
    new_tensor.set_gradient(Tensor::zeros(dim, None, None));
    new_tensor
}

pub fn log2<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    let dim: Dimensions = val.dim();
    let mut new_data = Vec::with_capacity(dim.0 * dim.1);
    let data: &DataArray<T> = val.data();
    for i in 0..dim.0 * dim.1 {
        new_data.push(data[i].log2());
    }
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    let mut new_tensor = Tensor::_build_raw(
        new_data,
        dim,
        None,
        Some(true),
        Some(Ops::UnaryOps(UnaryOps::LOG2)),
        Some(val),
        None
    );
    new_tensor.set_gradient(Tensor::zeros(dim, None, None));
    new_tensor
}

pub fn log<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    let new_val_base_two = log2(val);
    let e = T::from_f64(E).unwrap();
    let conversion_val = T::from_f64(2.0).unwrap().log(e);
    new_val_base_two * conversion_val
}

pub fn sum<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    let dim: Dimensions = val.dim();
    // container for new data
    let mut new_data = Vec::with_capacity(1);
    // get data
    let data: &DataArray<T> = val.data();
    // runnning sum
    let mut sum = T::zero();
    for i in 0..dim.0 * dim.1 {
        sum = sum + data[i];
    }
    new_data.push(sum);
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    let mut new_tensor = Tensor::_build_raw(
        new_data,
        (1, 1),
        None,
        Some(true),
        Some(Ops::ReduceOps(ReduceOps::SUM)),
        Some(val),
        None
    );
    new_tensor.set_gradient(Tensor::zeros(dim, None, None));
    new_tensor
}
