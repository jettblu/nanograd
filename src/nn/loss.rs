use crate::{ Tensor, TensorTrait };

pub fn categorical_cross_entropy<T: TensorTrait<T>>(
    y_pred: Tensor<T>,
    y_true: Tensor<T>
) -> Tensor<T> {
    panic!("Not implemented")
}
