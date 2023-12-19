use crate::{
    Tensor,
    TensorTrait,
    Ops,
    backward::binary::backward_binary,
    backward::unary::backward_unary,
    backward::reduce::backward_reduce,
};

pub fn backward_by_operation<T: TensorTrait<T>>(child: &mut Tensor<T>) {
    // control flow based on operation
    // get operation
    let parent = child.left.as_mut().unwrap();
    let op = child.op;
    let grad = child.gradient.as_mut().unwrap();

    match op {
        Ops::BinaryOps(_) => {
            println!("binary here!");
            let parent_2 = child.right.as_mut().unwrap();
            println!("binary here again!");
            backward_binary(parent, parent_2, grad, op);
        }
        Ops::ReduceOps(_) => {
            backward_reduce(parent, grad, op);
        }
        Ops::UnaryOps(_) => {
            backward_unary(parent, grad, op);
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
