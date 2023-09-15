use crate::{ Tensor, TensorTrait };

pub struct FeaturesAndLabels<T: TensorTrait<T>> {
    features: Tensor<T>,
    labels: Tensor<T>,
}

impl<T: TensorTrait<T>> FeaturesAndLabels<T> {
    pub fn new(features: Tensor<T>, labels: Tensor<T>) -> Self {
        FeaturesAndLabels{ features, labels }
    }
}
