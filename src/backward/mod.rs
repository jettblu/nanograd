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
    let grad: &mut Tensor<T>;
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
            // print grads
            println!("new_parent_grad_1: {:?}", new_parent_grad_1);
            println!("new_parent_grad_2: {:?}", new_parent_grad_2);
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
            // print grads
            println!("new_parent_grad_1: {:?}", new_parent_grad_1);
            println!("new_parent_grad_2: {:?}", new_parent_grad_2);
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
                    println!("Parents:");
                    println!("t.len(): {}", t.len());
                    parents = t;
                }
            }
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
            let parent_1_data: &DataArray<T> = parents[0].data();
            while i < dim.0 * dim.1 {
                // sigmoid derivative
                // sigmoid(x) * (1 - sigmoid(x))*curr grad + parent grad
                new_parent_grad_1.push(
                    parent_grad_1_data[i] +
                        grad_data[i] * parent_1_data[i] * (T::one() - parent_1_data[i])
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
        Ops::UnaryOps(UnaryOps::EXP2) => {}
        Ops::ReduceOps(ReduceOps::MAX) => {}
        Ops::ReduceOps(ReduceOps::SUM) => {}
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
