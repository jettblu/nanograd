use std::{ f64::consts::E, borrow::BorrowMut };

use crate::{
    Tensor,
    TensorTrait,
    Ops,
    types::ops::{ BinaryOps, UnaryOps, ReduceOps },
    DataArray,
    Dimensions,
};

pub fn backward_helper<T: TensorTrait<T>>(visited: &mut Vec<i32>, val: &mut Tensor<T>) {
    // requires gradient check handled by caller
    // run backward pass
    if !visited.contains(&val.unique_id) {
        // add to visited
        visited.push(val.unique_id);
        if val.op != Ops::None {
            backward_by_operation(val);
        }
        if val.prev.is_some() {
            let parents: &mut Box<Vec<Tensor<T>>>;
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            for parent in parents.iter_mut() {
                backward_helper(visited, parent);
            }
        }
    }
}

pub fn backward_by_operation<T: TensorTrait<T>>(val: &mut Tensor<T>) {
    // control flow based on operation
    // get operation
    let op = val.op;
    let parents: &mut Box<Vec<Tensor<T>>>;
    let grad: &mut Box<Tensor<T>>;
    match &mut val.gradient {
        None => {
            panic!("No gradient");
        }
        Some(t) => {
            grad = t;
        }
    }

    // get dimensions of gradient
    let dim: Dimensions = grad.dim();
    // get data
    let grad_data: &DataArray<T> = grad.data();
    match op {
        // addition case
        Ops::BinaryOps(BinaryOps::ADD) => {
            // gradient flows through addition
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            // iterate through grad
            let mut i: usize = 0;
            let parent_grad_1: &Box<Tensor<T>>;
            let parent_grad_2: &Box<Tensor<T>>;
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            match parents[1].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_2 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let mut new_parent_grad_2: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
            let parent_grad_2_data: &DataArray<T> = parent_grad_2.data();
            while i < dim.0 * dim.1 {
                new_parent_grad_1.push(parent_grad_1_data[i] + grad_data[i]);
                new_parent_grad_2.push(parent_grad_2_data[i] + grad_data[i]);
                i += 1;
            }
            // set gradients
            parents[0].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_1.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
            parents[1].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_2.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
        }
        // subtraction case
        Ops::BinaryOps(BinaryOps::SUB) => {
            // gradient flows through subtraction
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            // get dimensions of gradient
            let dim: Dimensions = grad.dim();
            // get data
            let grad_data: &DataArray<T> = grad.data();
            // iterate through grad
            let mut i: usize = 0;
            let parent_grad_1: &Box<Tensor<T>>;
            let parent_grad_2: &Box<Tensor<T>>;
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            match parents[1].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_2 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let mut new_parent_grad_2: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
            let parent_grad_2_data: &DataArray<T> = parent_grad_2.data();
            while i < dim.0 * dim.1 {
                new_parent_grad_1.push(parent_grad_1_data[i] - grad_data[i]);
                new_parent_grad_2.push(parent_grad_2_data[i] - grad_data[i]);
                i += 1;
            }
            // set gradients
            parents[0].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_1.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
            parents[1].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_2.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
        }
        // multiplication case
        Ops::BinaryOps(BinaryOps::MUL) => {
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            // iterate through grad
            let parent_grad_1: &Box<Tensor<T>>;
            let parent_grad_2: &Box<Tensor<T>>;
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            match parents[1].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_2 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let mut new_parent_grad_2: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
            let parent_grad_2_data: &DataArray<T> = parent_grad_2.data();
            let parent_1_data: &DataArray<T> = parents[0].data();
            let parent_2_data: &DataArray<T> = parents[1].data();
            // parent grad + curr grad * other parent data
            let other_dim = parents[0].dim();
            for i in 0..dim.0 {
                for j in 0..other_dim.1 {
                    let mut sum_1 = T::zero();
                    let mut sum_2 = T::zero();
                    for k in 0..other_dim.0 {
                        let index_a = i * dim.0 + k;
                        let index_b = k * other_dim.1 + j;
                        sum_1 = sum_1 + grad_data[index_a] * parent_2_data[index_b];
                        sum_2 = sum_2 + grad_data[index_a] * parent_1_data[index_a];
                    }
                    // can assume indices will be sequential
                    // curr index: i * other_dim.1 + j
                    new_parent_grad_1.push(parent_grad_1_data[i + j] + sum_1);
                    new_parent_grad_2.push(parent_grad_2_data[i + j] + sum_2);
                }
            }
            // set gradients
            parents[0].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_1.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
            parents[1].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_2.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
        }
        Ops::UnaryOps(UnaryOps::MAX) => {
            // 1s where max, 0s otherwise
            // gradient flows through max
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            // iterate through grad
            let mut i: usize = 0;
            let parent_grad_1: &Box<Tensor<T>>;
            let grad_data = grad.data();
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
            let zero = T::zero();
            while i < dim.0 * dim.1 {
                // max derivative
                // 1 if max, 0 otherwise
                let new_val = if parent_grad_1_data[i] != zero { T::one() } else { T::zero() };
                new_parent_grad_1.push(new_val * grad_data[i]);
                i += 1;
            }
        }
        // unary ops
        Ops::UnaryOps(UnaryOps::Sigmoid) => {
            // gradient flows through sigmoid
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            let curr_data = val.lazy_data.data();
            // iterate through grad
            let mut i: usize = 0;
            let parent_grad_1: &Box<Tensor<T>>;
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
            while i < dim.0 * dim.1 {
                // sigmoid derivative
                // sigmoid(x) * (1 - sigmoid(x))*curr grad + parent grad
                // TODO: update to use current node's data instead of recomputing sigmoid(x)
                let sigmoid: T = curr_data[i];
                new_parent_grad_1.push(
                    parent_grad_1_data[i] + grad_data[i] * sigmoid * (T::one() - sigmoid)
                );
                i += 1;
            }
            // set gradients
            parents[0].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_1.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
        }
        Ops::UnaryOps(UnaryOps::LOG2) => {
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            let curr_data = val.lazy_data.data();
            // iterate through grad
            let mut i: usize = 0;
            let parent_grad_1: &Box<Tensor<T>>;
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();

            while i < dim.0 * dim.1 {
                // log2 derivative
                // 1 / (x * ln(2)) * curr grad + parent grad
                let new_val =
                    parent_grad_1_data[i] +
                    grad_data[i] / (curr_data[i] * T::from_f64(E).unwrap().ln());
                new_parent_grad_1.push(new_val);
                i += 1;
            }
            // set gradients
            parents[0].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_1.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
        }
        Ops::UnaryOps(UnaryOps::EXP2) => {
            // gradient flows through exp2
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            let curr_data = val.lazy_data.data();
            // iterate through grad
            let mut i: usize = 0;
            let parent_grad_1: &Box<Tensor<T>>;
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
            while i < dim.0 * dim.1 {
                // exp2 derivative
                // exp2(x) * curr grad + parent grad
                let new_val = curr_data[i] * grad_data[i] + parent_grad_1_data[i];
                new_parent_grad_1.push(new_val);
                i += 1;
            }
            // set gradients
            parents[0].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_1.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
        }
        Ops::UnaryOps(UnaryOps::Softmax) => {
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            let curr_data = val.lazy_data.data();
            // iterate through grad
            let mut i: usize = 0;
            let parent_grad_1: &Box<Tensor<T>>;
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
            while i < dim.0 * dim.1 {
                // softmax derivative
                // softmax(x) * (1 - softmax(x)) * curr grad + parent grad
                let softmax: T = curr_data[i];
                new_parent_grad_1.push(
                    parent_grad_1_data[i] + grad_data[i] * softmax * (T::one() - softmax)
                );
                i += 1;
            }
        }
        Ops::ReduceOps(ReduceOps::MAX) => {
            panic!("Not implemented");
        }
        Ops::ReduceOps(ReduceOps::SUM) => {
            // gradient flows through sum
            match &mut val.prev {
                None => {
                    panic!("No parents");
                }
                Some(t) => {
                    parents = t;
                }
            }
            // iterate through grad
            let mut i: usize = 0;
            let parent_grad_1: &Box<Tensor<T>>;
            match parents[0].get_gradient() {
                None => {
                    panic!("No gradient");
                }
                Some(t) => {
                    parent_grad_1 = t;
                }
            }
            // get parent gradients
            let mut new_parent_grad_1: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
            let parent_grad_1_data: &DataArray<T> = parent_grad_1.data();
            while i < dim.0 * dim.1 {
                // sum derivative
                // 1 * curr grad + parent grad
                let new_val = grad_data[i] + parent_grad_1_data[i];
                new_parent_grad_1.push(new_val);
                i += 1;
            }
            // set gradients
            parents[0].set_gradient(
                Tensor::_build_raw(
                    new_parent_grad_1.into_boxed_slice(),
                    (dim.0.clone(), dim.1.clone()),
                    None,
                    None,
                    None,
                    None
                )
            );
        }
        Ops::UnaryOps(UnaryOps::SUM) => {
            panic!("Not implemented");
        }
        // shouldn't need to implement these
        Ops::TernaryOps(_) => {
            panic!("Not implemented");
        }
        Ops::LoadOps(_) => {
            panic!("Not implemented");
        }
        Ops::None => {
            panic!("Not implemented");
        }
    }
}
