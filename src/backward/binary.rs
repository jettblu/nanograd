use crate::{ Tensor, TensorTrait, Ops, types::ops::BinaryOps, DataArray, Dimensions };

pub fn backward_binary<T: TensorTrait<T>>(
    parent: &mut Tensor<T>,
    coparent: &mut Tensor<T>,
    child_gradient: &mut Tensor<T>,
    op: Ops
) {
    // get dimensions of gradient
    let dim: Dimensions = parent.dim();
    let dim_coparent: Dimensions = coparent.dim();
    // get data
    let parent_data: &DataArray<T> = parent.data();
    let coparent_data: &DataArray<T> = coparent.data();
    let child_gradient_data = child_gradient.data();
    let mut new_grad_parent: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
    let mut new_grad_coparent: Vec<T> = Vec::with_capacity(dim_coparent.0 * dim_coparent.1);
    // same as dim in previous version
    let dim = parent.dim();

    match op {
        // addition case
        Ops::BinaryOps(BinaryOps::ADD) => {
            // iterate through grad
            let mut i: usize = 0;

            while i < dim.0 * dim.1 {
                new_grad_parent.push(child_gradient_data[i]);
                new_grad_coparent.push(child_gradient_data[i]);
                i += 1;
            }
        }
        // subtraction case
        Ops::BinaryOps(BinaryOps::SUB) => {
            // iterate through grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                new_grad_parent.push(child_gradient_data[i]);
                new_grad_coparent.push(child_gradient_data[i]);
                i += 1;
            }
        }
        // multiplication case
        Ops::BinaryOps(BinaryOps::MUL) => {
            // iterate through grad
            let mut i: usize = 0;
            while i < dim.0 * dim.1 {
                // mul derivative
                let new_val_parent = child_gradient_data[i] * coparent_data[i];
                let new_val_coparent = child_gradient_data[i] * parent_data[i];
                new_grad_coparent.push(new_val_coparent);
                new_grad_parent.push(new_val_parent);
                i += 1;
            }
        }
        _ => {
            panic!("Not implemented");
        }
    }
    println!("binary returning");
    // set gradients
    parent.set_gradient(
        Tensor::_build_raw(
            new_grad_parent.into_boxed_slice(),
            (dim.0.clone(), dim.1.clone()),
            None,
            None,
            None,
            None,
            None
        )
    );
    coparent.set_gradient(
        Tensor::_build_raw(
            new_grad_coparent.into_boxed_slice(),
            (dim_coparent.0.clone(), dim_coparent.1.clone()),
            None,
            None,
            None,
            None,
            None
        )
    );
}

// pub fn backward_binary<T: TensorTrait<T>>(
//     val: &mut Tensor<T>,
//     grad: &TensorRef<T>,
//     sibling: &TensorRef<T>
// ) {
//     let op = val.op;
//     // get dimensions of gradient
//     let dim: Dimensions = grad.dim();
//     let grad_curr: &Box<Tensor<T>> = val.get_gradient().unwrap();
//     // get data
//     let grad_data: &DataArray<T> = grad.data();
//     let grad_curr_data: &DataArray<T> = val.data();
//     let curr_data: &DataArray<T> = val.data();
//     let sibling_data: &DataArray<T> = sibling.data();

//     let mut new_grad: Vec<T> = Vec::with_capacity(dim.0 * dim.1);
//     // same as other_dim in previous version
//     let dim_curr_grad = grad_curr.dim();
//     // same as dim in previous version
//     let dim = grad.dim();

//     match op {
//         // addition case
//         Ops::BinaryOps(BinaryOps::ADD) => {
//             // iterate through grad
//             let mut i: usize = 0;

//             while i < dim.0 * dim.1 {
//                 new_grad.push(grad_curr_data[i] + grad_data[i]);
//                 i += 1;
//             }
//         }
//         // subtraction case
//         Ops::BinaryOps(BinaryOps::SUB) => {
//             // iterate through grad
//             let mut i: usize = 0;
//             while i < dim.0 * dim.1 {
//                 new_grad.push(grad_curr_data[i] - grad_data[i]);
//                 i += 1;
//             }
//         }
//         // multiplication case
//         Ops::BinaryOps(BinaryOps::MUL) => {
//             // iterate through grad
//             let mut i: usize = 0;
//             while i < dim.0 * dim.1 {
//                 // mul derivative
//                 // parent grad + curr grad * other parent data
//                 let new_val = grad_data[i] * sibling_data[i] + grad_curr_data[i];
//                 new_grad.push(new_val);
//                 i += 1;
//             }
//         }
//         _ => {
//             panic!("Not implemented");
//         }
//     }
//     // set gradients
//     val.set_gradient(
//         Tensor::_build_raw(
//             new_grad.into_boxed_slice(),
//             (dim.0.clone(), dim.1.clone()),
//             None,
//             None,
//             None,
//             None,
//             None
//         )
//     );
// }

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
