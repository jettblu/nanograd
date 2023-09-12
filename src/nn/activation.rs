use core::panic;
use std::f32::consts::E;

use crate::{ TensorTrait, Tensor, Dimensions, DataArray, types::ops::UnaryOps, Ops };

use crate::nn::transformation::max;
use crate::nn::transformation::log;
/// Sigmoid function.
///
/// # Arguments
///
/// * `val` - The tensor to apply the sigmoid function to.
///
/// # Returns
///
/// A tensor with the sigmoid function applied to it element-wise.
pub fn sigmoid<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    let dim: Dimensions = val.dim();
    let mut new_data = Vec::with_capacity(dim.0 * dim.1);
    let val_data = val.data();
    let exp_typed = T::from_f32(E);
    let exp_typed: T = match exp_typed {
        Some(exp_typed) => exp_typed,
        None => panic!("Error converting E to T"),
    };
    let one = T::one();
    for i in 0..dim.0 * dim.1 {
        new_data.push(exp_typed.pow(val_data[i]) / (one + exp_typed.pow(val_data[i])));
        println!("new_data[{}] = {}", i, new_data[i]);
    }
    // create Box<[T]> from Vec<T>
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    let mut new_tensor = Tensor::_build_raw(
        new_data,
        dim,
        None,
        Some(true),
        Some(Ops::UnaryOps(UnaryOps::Sigmoid)),
        Some(vec![val])
    );
    new_tensor.set_gradient(Tensor::zeros(dim, None, None));
    new_tensor
}

// relu
pub fn relu<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    max(val, T::zero())
}

/// Hyperbolic tangent function.
///
/// # Arguments
///
/// * `val` - The tensor to apply the hyperbolic tangent function to.
///
/// # Returns
///
/// A tensor with the hyperbolic tangent function applied to it element-wise.
///
pub fn tanh<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    let one: T = T::one();
    let two: T = one + one;
    sigmoid(val * two) * two - one
}

///
/// Applies the softmax function to the tensor.
///
/// # Arguments
///
/// * `val` - The tensor to apply the softmax function to.
///
/// # Returns
///
/// A tensor with the softmax function applied to it. New tensor has the same shape as the input tensor.
///
/// # Examples
///
/// ```
/// use nanograd::{ Tensor, nn::activation::softmax };
///
/// let data = vec![1.0, 2.0, 3.0, 9.0].into_boxed_slice();
///
/// let tensor = Tensor::new(data, (2, 2), None, Some(true));
///
/// let tensor_softmax = softmax(tensor);
///
/// let expected_result = &vec![0.26894142734067883, 0.7310585726593212, 0.0024726236060504682, 0.9975273763939495].into_boxed_slice();
///
///
/// assert_eq!(tensor_softmax.data(), expected_result);
///
/// ```
///

pub fn softmax<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    let dim: Dimensions = val.dim();
    let mut new_data = Vec::with_capacity(dim.0 * dim.1);
    let val_data = val.data();
    let exp_typed = T::from_f32(E);
    let exp_typed: T = match exp_typed {
        Some(exp_typed) => exp_typed,
        None => panic!("Error converting E to T"),
    };
    // iterate through each row of the tensor
    for i in 0..dim.0 {
        // get the sum of the row
        let mut sum: T = T::zero();
        for j in 0..dim.1 {
            sum = sum + exp_typed.pow(val_data[i * dim.1 + j]);
        }
        // iterate through each element in the row
        for j in 0..dim.1 {
            new_data.push(exp_typed.pow(val_data[i * dim.1 + j]) / sum);
        }
    }
    // create Box<[T]> from Vec<T>
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    let mut new_tensor = Tensor::_build_raw(
        new_data,
        dim,
        None,
        Some(true),
        Some(Ops::UnaryOps(UnaryOps::EXP2)),
        Some(vec![val])
    );
    new_tensor.set_gradient(Tensor::zeros(dim, None, None));
    new_tensor
}

pub fn log_softmax<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    let x = softmax(val);
    log(x)
}
