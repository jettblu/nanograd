use std::f64::consts::E;

use crate::{
    Tensor,
    TensorTrait,
    Ops,
    types::ops::UnaryOps,
    DataArray,
    Dimensions,
    tensor::TensorRef,
};

pub fn forward_unary<T: TensorTrait<T>>(parent: &mut Tensor<T>, child_grad: &TensorRef<T>) {
    let op = parent.op;
    // get dimensions of gradient
    let dim: Dimensions = child_grad.dim();
    // get data
    let grad_child_data: &DataArray<T> = child_grad.data();
    let grad_parent_data: &DataArray<T> = parent.data();
    let parent_data: &DataArray<T> = parent.data();

    let mut new_grad: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
    match op {
        Ops::UnaryOps(UnaryOps::MAX) => {
            // 1s where max, 0s otherwise
            // gradient flows through max
            // iterate through child_grad
            let mut i: usize = 0;
            let zero = T::zero();
            while i < dim.0 * dim.1 {
                // max derivative
                // 1 if max, 0 otherwise
                let new_val = if grad_child_data[i] != zero { T::one() } else { T::zero() };
                new_grad.push(new_val * grad_parent_data[i]);
                i += 1;
            }
        }
        // unary ops
        Ops::UnaryOps(UnaryOps::Sigmoid) => {
            // iterate through child_grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // sigmoid derivative
                // sigmoid(x) * (1 - sigmoid(x))*curr child_grad + parent child_grad
                // TODO: update to use current node's data instead of recomputing sigmoid(x)
                let sigmoid: T = parent_data[i];
                new_grad.push(
                    grad_child_data[i] + grad_parent_data[i] * sigmoid * (T::one() - sigmoid)
                );
                i += 1;
            }
        }
        Ops::UnaryOps(UnaryOps::LOG2) => {
            // iterate through child_grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // log2 derivative
                // 1 / (x * ln(2)) * curr child_grad + parent child_grad
                let new_val =
                    grad_parent_data[i] +
                    grad_child_data[i] / (parent_data[i] * T::from_f64(E).unwrap().ln());
                new_grad.push(new_val);
                i += 1;
            }
        }
        Ops::UnaryOps(UnaryOps::EXP2) => {
            // iterate through child_grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // exp2 derivative
                // exp2(x) * curr child_grad + parent child_grad
                let new_val = parent_data[i] * grad_parent_data[i] + grad_child_data[i];
                new_grad.push(new_val);
                i += 1;
            }
        }
        Ops::UnaryOps(UnaryOps::Softmax) => {
            // iterate through child_grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // softmax derivative
                // softmax(x) * (1 - softmax(x)) * curr child_grad + parent child_grad
                let softmax: T = parent_data[i];
                new_grad.push(
                    grad_child_data[i] + grad_parent_data[i] * softmax * (T::one() - softmax)
                );
                i += 1;
            }
        }
        _ => {
            panic!("Not implemented");
        }
    }
    // set gradients
    parent.set_gradient(
        Tensor::_build_raw(
            new_grad.into_boxed_slice(),
            (dim.0.clone(), dim.1.clone()),
            None,
            None,
            None,
            None,
            None
        )
    );
}
