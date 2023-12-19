use crate::{ TensorTrait, Dimensions, helpers::is_valid_matrix_multiplication, DataArray };

pub fn mul_data<T: TensorTrait<T>>(
    a_data: &Box<[T]>,
    a_dim: Dimensions,
    b_data: &Box<[T]>,
    b_dim: Dimensions
) -> DataArray<T> {
    if !is_valid_matrix_multiplication(a_dim, b_dim) {
        panic!("Invalid matrix multiplication");
    }
    let mut new_data = Vec::with_capacity(a_dim.0 * b_dim.1);
    // multiply vals
    for i in 0..a_dim.0 {
        for j in 0..b_dim.1 {
            let mut sum = T::zero();
            for k in 0..b_dim.0 {
                let index_a = i * a_dim.0 + k;
                let index_b = k * b_dim.1 + j;
                sum = sum + a_data[index_a] * b_data[index_b];
            }
            // can assume indices will be sequential
            // curr index: i * other_dim.1 + j
            new_data.push(sum);
        }
    }
    return new_data.into_boxed_slice();
}

pub fn add_data<T: TensorTrait<T>>(
    a_data: &Box<[T]>,
    a_dim: Dimensions,
    b_data: &Box<[T]>,
    b_dim: Dimensions
) -> DataArray<T> {
    if a_dim != b_dim {
        panic!("Invalid addition");
    }
    let mut new_data = Vec::with_capacity(a_dim.0 * a_dim.1);
    for i in 0..a_dim.0 {
        for j in 0..a_dim.1 {
            let index = i * a_dim.0 + j;
            new_data.push(a_data[index] + b_data[index]);
        }
    }
    return new_data.into_boxed_slice();
}

pub fn subtract_data<T: TensorTrait<T>>(
    a_data: &Box<[T]>,
    a_dim: Dimensions,
    b_data: &Box<[T]>,
    b_dim: Dimensions
) -> DataArray<T> {
    if a_dim != b_dim {
        panic!("Invalid subtraction");
    }
    let mut new_data = Vec::with_capacity(a_dim.0 * a_dim.1);
    for i in 0..a_dim.0 {
        for j in 0..a_dim.1 {
            let index = i * a_dim.0 + j;
            new_data.push(a_data[index] - b_data[index]);
        }
    }
    return new_data.into_boxed_slice();
}
