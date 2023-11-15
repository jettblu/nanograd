use crate::{
    Tensor,
    TensorTrait,
    Ops,
    types::ops::ReduceOps,
    DataArray,
    Dimensions,
    tensor::TensorRef,
};

pub fn backward_reduce<T: TensorTrait<T>>(val: &mut Tensor<T>, grad: &TensorRef<T>) {
    let op = val.op;
    // get dimensions of gradient
    let dim: Dimensions = grad.dim();
    // get data
    let grad_data: &DataArray<T> = grad.data();
    let grad_curr_data: &DataArray<T> = val.data();

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
                // 1 * curr grad + parent grad
                let new_val = grad_curr_data[i] + grad_data[i];
                new_grad.push(new_val);
                i += 1;
            }
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
            )
        }
        _ => {
            panic!("Not implemented");
        }
    }
}
