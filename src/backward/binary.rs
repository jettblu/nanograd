use crate::{
    Tensor,
    TensorTrait,
    Ops,
    types::ops::BinaryOps,
    DataArray,
    Dimensions,
    tensor::TensorRef,
};

pub fn backward_binary<T: TensorTrait<T>>(
    val: &mut Tensor<T>,
    grad: &TensorRef<T>,
    sibling: &TensorRef<T>
) {
    let op = val.op;
    // get dimensions of gradient
    let dim: Dimensions = grad.dim();
    let grad_curr: &Box<Tensor<T>> = val.get_gradient().unwrap();
    // get data
    let grad_data: &DataArray<T> = grad.data();
    let grad_curr_data: &DataArray<T> = val.data();
    let curr_data: &DataArray<T> = val.data();
    let sibling_data: &DataArray<T> = sibling.data();

    let mut new_grad: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
    // same as other_dim in previous version
    let dim_curr_grad = grad_curr.dim();
    // same as dim in previous version
    let dim = grad.dim();

    match op {
        // addition case
        Ops::BinaryOps(BinaryOps::ADD) => {
            // iterate through grad
            let mut i: usize = 0;

            while i < dim.0 * dim.1 {
                new_grad.push(grad_curr_data[i] + grad_data[i]);
                i += 1;
            }
        }
        // subtraction case
        Ops::BinaryOps(BinaryOps::SUB) => {
            // iterate through grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                new_grad.push(grad_curr_data[i] - grad_data[i]);
                i += 1;
            }
        }
        // multiplication case
        Ops::BinaryOps(BinaryOps::MUL) => {
            // iterate through grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // mul derivative
                // parent grad + curr grad * other parent data
                let new_val = grad_data[i] * sibling_data[i] + grad_curr_data[i];
                new_grad.push(new_val);
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

// pub fn backward_binary<T: TensorTrait<T>>(val: TensorRef<T>) {
//     let op = val.op;
//     // get gradient for value
//     let grad: &mut Box<Tensor<T>>;
//     match &mut val.gradient {
//         None => {
//             panic!("No gradient");
//         }
//         Some(t) => {
//             grad = t;
//         }
//     }
//     // ensure we have both parents
//     if !val.left.is_some() || !val.right.is_some() {
//         panic!("No left or right");
//     }
//     // get dimensions of gradient
//     let dim: Dimensions = grad.dim();
//     let dim_parent: Dimensions = val.dim().borrow_mut();
//     // get data
//     let grad_data: &DataArray<T> = grad.data();
//     let parent_grad_1: &Box<Tensor<T>> = val.left.as_ref().unwrap();
//     let parent_grad_2: &Box<Tensor<T>> = val.right.as_ref().unwrap();
//     // get parent gradients
//     let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
//     let mut new_parent_grad_2: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
//     let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
//     let parent_grad_2_data: &DataArray<T> = parent_grad_2.data();
//     // parent grad + curr grad * other parent data
//     let other_dim = parent_grad_1.dim();
