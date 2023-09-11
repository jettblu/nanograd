use core::panic;
use std::f32::consts::E;

use crate::{ TensorTrait, Tensor, Dimensions, DataArray, types::ops::UnaryOps, Ops };

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
    val.max(T::zero())
}

pub fn leaky_relu<T: TensorTrait<T>>(val: Tensor<T>) -> Tensor<T> {
    // not implemented for now since backward op is not defined
    panic!("Not implemented");
    let alpha: Option<T> = T::from_f32(-0.01);
    let alpha: T = match alpha {
        Some(alpha) => alpha,
        None => panic!("Error converting alpha to T"),
    };
    let relu_1 = val.max(T::zero());
    let val_signs_flipped = val * alpha;
    let output = relu_1 + val_signs_flipped;
    output
}
