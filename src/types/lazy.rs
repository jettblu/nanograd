use std::ops::{ Add, Sub, Mul, Neg };

use crate::{
    TensorTrait,
    Device,
    default_device,
    is_valid_matrix_multiplication,
    Ops,
    types::ops::BinaryOps,
    new_dimensions_after_matrix_multiplication,
};

pub struct LazyBuffer<T: TensorTrait<T>> {
    data: DataArray<T>,
    dimensions: Dimensions,
    device: Device,
    realized: bool,
    op: Option<Ops>,
    // prev
}

impl<T> LazyBuffer<T> where T: TensorTrait<T> {
    pub fn new(
        data: DataArray<T>,
        dimensions: Dimensions,
        device: Option<Device>,
        op: Option<Ops>
    ) -> Self {
        let device = match device {
            Some(device) => device,
            None => default_device(),
        };
        Self { data, dimensions, device, realized: false, op }
    }

    pub fn dim(&self) -> Dimensions {
        self.dimensions
    }
    pub fn data(&self) -> &DataArray<T> {
        &self.data
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn realize(&self) -> &LazyBuffer<T> {
        // TODO: implement
        // for now just return data since always realized
        self
    }
}

// implement add for LazyBuffer
fn add<T: TensorTrait<T>>(a: &LazyBuffer<T>, b: &LazyBuffer<T>) -> LazyBuffer<T> {
    // make sure dimensions match
    let a_dim: Dimensions = a.dim();
    let b_dim: Dimensions = b.dim();
    // can only add LazyBuffers of same dimensions
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
    LazyBuffer::new(new_data, a_dim, None, Some(Ops::BinaryOps(BinaryOps::ADD)))
}

fn mul<T: TensorTrait<T>>(a: &LazyBuffer<T>, b: &LazyBuffer<T>) -> LazyBuffer<T> {
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
    LazyBuffer::new(new_data, new_dim, None, Some(Ops::BinaryOps(BinaryOps::MUL)))
}

fn neg<T: TensorTrait<T>>(a: &LazyBuffer<T>) -> LazyBuffer<T> {
    a * &(T::zero() - T::one())
}

fn sub<T: TensorTrait<T>>(a: &LazyBuffer<T>, b: &LazyBuffer<T>) -> LazyBuffer<T> {
    // make sure dimensions match
    let a_dim: Dimensions = a.dim();
    let b_dim: Dimensions = b.dim();
    // can only add LazyBuffers of same dimensions
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
    LazyBuffer::new(new_data, a_dim, None, None)
}

// math operations
impl<T> Add<LazyBuffer<T>> for LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn add(self, other: LazyBuffer<T>) -> LazyBuffer<T> {
        add(&self, &other)
    }
}

// support adding ref to ref
impl<'a, T> Add<&'a LazyBuffer<T>> for &'a LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn add(self, other: &'a LazyBuffer<T>) -> LazyBuffer<T> {
        add(self, other)
    }
}

// supoort subtraction
impl<T> Sub<LazyBuffer<T>> for LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn sub(self, other: LazyBuffer<T>) -> LazyBuffer<T> {
        sub(&self, &other)
    }
}

// support negation
impl<T> Neg for LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn neg(self) -> LazyBuffer<T> {
        neg(&self)
    }
}

// support subtracting ref to ref
impl<'a, T> Sub<&'a LazyBuffer<T>> for &'a LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn sub(self, other: &'a LazyBuffer<T>) -> LazyBuffer<T> {
        sub(self, other)
    }
}

// support multiplying buffer by T
impl<T> Mul<T> for LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn mul(self, other: T) -> LazyBuffer<T> {
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
        LazyBuffer::new(new_data, dim, None, Some(Ops::BinaryOps(BinaryOps::MUL)))
    }
}

// support multiplying buffer ref by T ref
impl<'a, T> Mul<&'a T> for &'a LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn mul(self, other: &'a T) -> LazyBuffer<T> {
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
        LazyBuffer::new(new_data, dim, None, Some(Ops::BinaryOps(BinaryOps::MUL)))
    }
}

// support mutltipying
impl<T> Mul<LazyBuffer<T>> for LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn mul(self, other: LazyBuffer<T>) -> LazyBuffer<T> {
        mul(&self, &other)
    }
}

// supporting multiplying ref to ref
impl<'a, T> Mul<&'a LazyBuffer<T>> for &'a LazyBuffer<T> where T: TensorTrait<T> {
    type Output = LazyBuffer<T>;
    fn mul(self, other: &'a LazyBuffer<T>) -> LazyBuffer<T> {
        mul(self, other)
    }
}

pub type DataArray<T> = Box<[T]>;

pub type Dimensions = (usize, usize);
