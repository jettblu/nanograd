use crate::{
    Tensor,
    TensorTrait,
    Ops,
    types::ops::ReduceOps,
    DataArray,
    Dimensions,
    tensor::TensorRef,
};

pub fn forward_reduce<T: TensorTrait<T>>(parent: &mut Tensor<T>, child_grad: &TensorRef<T>) {
    let op = parent.op;
    // get dimensions of gradient
    let dim: Dimensions = child_grad.dim();
    // get data
    let grad_child_data: &DataArray<T> = child_grad.data();
    let grad_parent_data: &DataArray<T> = parent.data();

    let mut new_grad: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
    match op {
        Ops::ReduceOps(ReduceOps::MAX) => {
            panic!("Not implemented");
        }
        Ops::ReduceOps(ReduceOps::SUM) => {
            let mut i: usize = 0;
            // gradient flows throwugh sum
            while i < dim.0 * dim.1 {
                // sum derivative
                // 1 * curr child_grad + parent child_grad
                let new_val = grad_parent_data[i] + grad_child_data[i];
                new_grad.push(new_val);
                i += 1;
            }
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
            )
        }
        _ => {
            panic!("Not implemented");
        }
    }
}
