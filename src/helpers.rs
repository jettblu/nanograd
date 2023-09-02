use crate::Dimensions;

pub fn is_valid_matrix_multiplication(a_dim: Dimensions, b_dim: Dimensions) -> bool {
    if a_dim.1 == b_dim.0 {
        return true;
    }
    false
}

pub fn new_dimensions_after_matrix_multiplication(
    a_dim: Dimensions,
    b_dim: Dimensions
) -> Dimensions {
    (a_dim.0, b_dim.1)
}
