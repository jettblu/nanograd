use std::ops::Add;

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
