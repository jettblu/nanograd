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

pub fn backward_unary<T: TensorTrait<T>>(val: &mut Tensor<T>, grad: &TensorRef<T>) {
    let op = val.op;
    // get dimensions of gradient
    let dim: Dimensions = grad.dim();
    // get data
    let grad_data: &DataArray<T> = grad.data();
    let grad_curr_data: &DataArray<T> = val.data();
    let curr_data: &DataArray<T> = val.data();

    let mut new_grad: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
    match op {
        Ops::UnaryOps(UnaryOps::MAX) => {
            // 1s where max, 0s otherwise
            // gradient flows through max
            // iterate through grad
            let mut i: usize = 0;
            let zero = T::zero();
            while i < dim.0 * dim.1 {
                // max derivative
                // 1 if max, 0 otherwise
                let new_val = if grad_data[i] != zero { T::one() } else { T::zero() };
                new_grad.push(new_val * grad_curr_data[i]);
                i += 1;
            }
        }
        // unary ops
        Ops::UnaryOps(UnaryOps::Sigmoid) => {
            // iterate through grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // sigmoid derivative
                // sigmoid(x) * (1 - sigmoid(x))*curr grad + parent grad
                // TODO: update to use current node's data instead of recomputing sigmoid(x)
                let sigmoid: T = curr_data[i];
                new_grad.push(grad_data[i] + grad_curr_data[i] * sigmoid * (T::one() - sigmoid));
                i += 1;
            }
        }
        Ops::UnaryOps(UnaryOps::LOG2) => {
            // iterate through grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // log2 derivative
                // 1 / (x * ln(2)) * curr grad + parent grad
                let new_val =
                    grad_curr_data[i] +
                    grad_data[i] / (curr_data[i] * T::from_f64(E).unwrap().ln());
                new_grad.push(new_val);
                i += 1;
            }
        }
        Ops::UnaryOps(UnaryOps::EXP2) => {
            // iterate through grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // exp2 derivative
                // exp2(x) * curr grad + parent grad
                let new_val = curr_data[i] * grad_curr_data[i] + grad_data[i];
                new_grad.push(new_val);
                i += 1;
            }
        }
        Ops::UnaryOps(UnaryOps::Softmax) => {
            // iterate through grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // softmax derivative
                // softmax(x) * (1 - softmax(x)) * curr grad + parent grad
                let softmax: T = curr_data[i];
                new_grad.push(grad_data[i] + grad_curr_data[i] * softmax * (T::one() - softmax));
                i += 1;
            }
        }
        _ => {
            panic!("Not implemented");
        }
    }
    // set gradients
    val.set_gradient(
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
