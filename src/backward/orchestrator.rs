use std::{ f64::consts::E, borrow::BorrowMut };

use crate::{
    Tensor,
    TensorTrait,
    Ops,
    backward::binary::backward_binary,
    backward::unary::backward_unary,
    backward::reduce::backward_reduce,
    tensor::TensorRef,
    types::ops::BinaryOps,
};

pub fn backward_by_operation<T: TensorTrait<T>>(
    val: &mut Tensor<T>,
    grad: &TensorRef<T>,
    sibling: Option<&TensorRef<T>>
) {
    // control flow based on operation
    // get operation
    let op = val.op;
    match op {
        Ops::BinaryOps(_) => {
            let sibling_node = sibling.unwrap();
            backward_binary(val, grad, sibling_node);
        }
        Ops::ReduceOps(_) => {
            backward_reduce(val, grad);
        }
        Ops::UnaryOps(_) => {
            backward_unary(val, grad);
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
