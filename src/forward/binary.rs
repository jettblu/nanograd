use crate::{
    Tensor,
    TensorTrait,
    Ops,
    types::ops::BinaryOps,
    DataArray,
    Dimensions,
    forward::utils::{ mul_data, add_data, subtract_data },
};

pub fn forward_binary<T: TensorTrait<T>>(
    parent: &mut Tensor<T>,
    coparent: &mut Tensor<T>,
    child: &mut Tensor<T>,
    op: Ops
) {
    // get dimensions of gradient
    let dim: Dimensions = parent.dim();
    let dim_coparent: Dimensions = coparent.dim();

    let parent_data: &DataArray<T> = parent.data();
    let coparent_data: &DataArray<T> = coparent.data();

    match op {
        // addition case
        Ops::BinaryOps(BinaryOps::ADD) => {
            // iterate through grad
            let new_data: DataArray<T> = add_data(parent_data, dim, coparent_data, dim_coparent);
            child.set_data(new_data);
        }
        // subtraction case
        Ops::BinaryOps(BinaryOps::SUB) => {
            let new_data: DataArray<T> = subtract_data(
                parent_data,
                dim,
                coparent_data,
                dim_coparent
            );
            child.set_data(new_data);
        }
        // multiplication case
        Ops::BinaryOps(BinaryOps::MUL) => {
            let new_data: DataArray<T> = mul_data(parent_data, dim, coparent_data, dim_coparent);
            child.set_data(new_data);
        }
        _ => {
            panic!("Not implemented");
        }
    }
    println!("binary returning");
}
