use crate::{ Tensor, TensorTrait };

pub struct DataAndLabels<T: TensorTrait<T>> {
    data: Tensor<T>,
    labels: Tensor<T>,
}

impl<T: TensorTrait<T>> DataAndLabels<T> {
    pub fn new(data: Tensor<T>, labels: Tensor<T>) -> Self {
        DataAndLabels { data, labels }
    }
}
