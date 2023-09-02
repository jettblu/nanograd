use std::fmt;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use std::vec;

use crate::DataArray;
use crate::Device;
use crate::Dimensions;
use crate::LazyBuffer;
use crate::TensorTrait;
use crate::is_valid_matrix_multiplication;
use crate::new_dimensions_after_matrix_multiplication;

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
        add(&self, &other)
    }
}

// support adding ref to ref
impl<'a, T> Add<&'a Tensor<T>> for &'a Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn add(self, other: &'a Tensor<T>) -> Tensor<T> {
        add(self, other)
    }
}

// supoort subtraction
impl<T> Sub<Tensor<T>> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn sub(self, other: Tensor<T>) -> Tensor<T> {
        sub(&self, &other)
    }
}

// support subtracting ref to ref
impl<'a, T> Sub<&'a Tensor<T>> for &'a Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn sub(self, other: &'a Tensor<T>) -> Tensor<T> {
        sub(self, other)
    }
}

impl<T> Mul<T> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn mul(self, other: T) -> Tensor<T> {
        let dim: Dimensions = self.dim();
        let mut new_data = Vec::with_capacity(dim.0 * dim.1);
        let mut i: usize = 0;
        let self_data = self.data();
        while i < dim.0 * dim.1 {
            new_data.push(self_data[i] * other);
            i += 1;
        }
        // create Box<[T]> from Vec<T>
        let new_data: DataArray<T> = new_data.into_boxed_slice();
        Tensor::new(new_data, dim, None, None)
    }
}

// support multiplying ref to ref
impl<'a, T> Mul<&'a T> for &'a Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn mul(self, other: &'a T) -> Tensor<T> {
        let dim: Dimensions = self.dim();
        let mut new_data = Vec::with_capacity(dim.0 * dim.1);
        let mut i: usize = 0;
        let self_data = self.data();
        while i < dim.0 * dim.1 {
            new_data.push(self_data[i] * *other);
            i += 1;
        }
        // create Box<[T]> from Vec<T>
        let new_data: DataArray<T> = new_data.into_boxed_slice();
        Tensor::new(new_data, dim, None, None)
    }
}

// support mutltipying
impl<T> Mul<Tensor<T>> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        mul(&self, &other)
    }
}

// supporting multiplying ref to ref
impl<'a, T> Mul<&'a Tensor<T>> for &'a Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn mul(self, other: &'a Tensor<T>) -> Tensor<T> {
        mul(self, other)
    }
}

// implement display trait for lazy buffer
impl<T> fmt::Display for Tensor<T> where T: TensorTrait<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // diplay array as matrix
        let dim: Dimensions = self.dim();
        let mut i: usize = 0;
        write!(f, "Tensor ({:} by {:}):\n", dim.0, dim.1)?;
        while i < dim.0 {
            let mut j: usize = 0;
            while j < dim.1 {
                let index = i * dim.0 + j;
                write!(f, "{:} ", self.data()[index])?;
                j += 1;
            }
            write!(f, "\n")?;
            i += 1;
        }
        write!(f, "\n")
    }
}

fn add<T: TensorTrait<T>>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    // make sure dimensions match
    let a_dim: Dimensions = a.dim();
    let b_dim: Dimensions = b.dim();
    // can only add tensors of same dimensions
    assert_eq!(a_dim, b_dim);
    let array_len = a_dim.0 * a_dim.1;
    let mut new_data = Vec::with_capacity(a_dim.0 * a_dim.1);
    let mut i: usize = 0;
    let a_data = a.data();
    let b_data = b.data();
    while i < array_len {
        new_data.push(a_data[i] + b_data[i]);
        i += 1;
    }
    // create Box<[T]> from Vec<T>
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    Tensor::new(new_data, a_dim, None, None)
}

fn mul<T: TensorTrait<T>>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    // make sure dimensions match
    let a_dim: Dimensions = a.dim();
    let b_dim: Dimensions = b.dim();
    if !is_valid_matrix_multiplication(a_dim, b_dim) {
        panic!("Invalid matrix multiplication");
    }
    let mut new_data = Vec::with_capacity(a_dim.0 * b_dim.1);
    // print new data len
    println!("new_data.len(): {}", new_data.len());
    let a_data = a.data();
    let b_data = b.data();
    // multiply vals
    for i in 0..a_dim.0 {
        for j in 0..b_dim.1 {
            println!("i: {}, j: {}", i, j);
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
    // create Box<[T]> from Vec<T>
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    let new_dim: Dimensions = new_dimensions_after_matrix_multiplication(a_dim, b_dim);
    Tensor::new(new_data, new_dim, None, None)
}

fn neg<T: TensorTrait<T>>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    // make sure dimensions match
    let a_dim: Dimensions = a.dim();
    let b_dim: Dimensions = b.dim();
    // can only add tensors of same dimensions
    assert_eq!(a_dim, b_dim);
    let array_len = a_dim.0 * a_dim.1;
    let mut new_data = Vec::with_capacity(a_dim.0 * a_dim.1);
    let mut i: usize = 0;
    let a_data = a.data();
    let b_data = b.data();
    while i < array_len {
        new_data.push(a_data[i] - b_data[i]);
        i += 1;
    }
    // create Box<[T]> from Vec<T>
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    Tensor::new(new_data, a_dim, None, None)
}

fn sub<T: TensorTrait<T>>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    // make sure dimensions match
    let a_dim: Dimensions = a.dim();
    let b_dim: Dimensions = b.dim();
    // can only add tensors of same dimensions
    assert_eq!(a_dim, b_dim);
    let array_len = a_dim.0 * a_dim.1;
    let mut new_data = Vec::with_capacity(a_dim.0 * a_dim.1);
    let mut i: usize = 0;
    let a_data = a.data();
    let b_data = b.data();
    let neg_one = T::zero() - T::one();
    while i < array_len {
        new_data.push(a_data[i] + neg_one * b_data[i]);
        i += 1;
    }
    // create Box<[T]> from Vec<T>
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    Tensor::new(new_data, a_dim, None, None)
}
