use std::ops::Add;
use std::vec;

use crate::DataArray;
use crate::Device;
use crate::Dimensions;
use crate::LazyBuffer;
use crate::TensorTrait;

pub struct Tensor<T: TensorTrait<T>> {
    lazy_data: LazyBuffer<T>,
    requires_grad: bool,
}

impl<T> Tensor<T> where T: TensorTrait<T> {
    // new tensor
    pub fn new(
        data: DataArray<T>,
        dimensions: Dimensions,
        device: Option<Device>,
        requires_grad: Option<bool>
    ) -> Self {
        let requires_grad = match requires_grad {
            Some(requires_grad) => requires_grad,
            None => false,
        };
        let new_array: [[u8; 10]; 20] = [[0; 10]; 20];
        let lazy_data: LazyBuffer<T> = LazyBuffer::new(data, dimensions, device);
        Self { lazy_data, requires_grad }
    }
    // get dimensions
    pub fn dim(&self) -> Dimensions {
        self.lazy_data.dim()
    }
    // get data
    pub fn data(&self) -> &DataArray<T> {
        self.lazy_data.data()
    }
    // get device
    pub fn device(&self) -> &Device {
        self.lazy_data.device()
    }
    // get requires_grad
    pub fn requires_grad(&self) -> &bool {
        &self.requires_grad
    }
    pub fn full(
        dim: Dimensions,
        fill_value: T,
        device: Option<Device>,
        requires_grad: Option<bool>
    ) -> Self {
        let data: DataArray<T> = vec![fill_value; dim.0 * dim.1].into_boxed_slice();
        Self::new(data, dim, device, requires_grad)
    }

    pub fn zeros(dim: Dimensions, device: Option<Device>, requires_grad: Option<bool>) -> Self {
        Self::full(dim, T::zero(), device, requires_grad)
    }

    pub fn ones(dim: Dimensions, device: Option<Device>, requires_grad: Option<bool>) -> Self {
        Self::full(dim, T::one(), device, requires_grad)
    }

    pub fn full_like(other: Tensor<T>, fill_value: T) -> Self {
        let dim: Dimensions = other.dim();
        let device: Option<Device> = Some(other.device().clone());
        let requires_grad: Option<bool> = Some(other.requires_grad().clone());
        Self::full(dim, fill_value, device, requires_grad)
    }

    pub fn zeros_like(other: Tensor<T>) -> Self {
        Self::full_like(other, T::zero())
    }

    pub fn ones_like(other: Tensor<T>) -> Self {
        Self::full_like(other, T::one())
    }
}

impl<T> Add<Tensor<T>> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn add(self, other: Tensor<T>) -> Tensor<T> {
        // make sure dimensions match
        let self_dim: Dimensions = self.dim();
        let other_dim: Dimensions = other.dim();
        assert_eq!(self.dim(), other.dim());
        let array_len = self_dim.0 * self_dim.1;
        let mut new_data = Vec::with_capacity(self_dim.0 * self_dim.1);
        let mut i: usize = 0;
        while i < array_len {
            new_data.push(self.data()[i] + other.data()[i]);
            i += 1;
        }
        // create Box<[T]> from Vec<T>
        let new_data: DataArray<T> = new_data.into_boxed_slice();
        Tensor::new(new_data, self_dim, None, None)
    }
}
