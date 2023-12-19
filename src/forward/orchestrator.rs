use crate::{
    Tensor,
    TensorTrait,
    Ops,
    forward::binary::forward_binary,
    forward::unary::forward_unary,
    forward::reduce::forward_reduce,
};

pub fn forward_by_operation<T: TensorTrait<T>>(child: &mut Tensor<T>) {
    // control flow based on operation
    // get operation
    let parent = child.left.as_mut().unwrap();
    let op = child.op;
    let grad = child.gradient.as_mut().unwrap();

    match op {
        Ops::BinaryOps(_) => {
            let parent_2 = child.right.as_mut().unwrap();
            forward_binary(parent, parent_2, grad, op);
        }
        Ops::ReduceOps(_) => {
            forward_reduce(parent, grad);
        }
        Ops::UnaryOps(_) => {
            forward_unary(parent, grad);
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
