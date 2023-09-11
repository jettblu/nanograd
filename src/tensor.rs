use core::panic;
use std::fmt;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use std::vec;
use std::hash::Hash;

use crate::DataArray;
use crate::Device;
use crate::Dimensions;
use crate::LazyBuffer;
use crate::Ops;
use crate::TensorTrait;
use crate::backward_helper;
use crate::is_valid_matrix_multiplication;
use crate::new_dimensions_after_matrix_multiplication;
use crate::random_number;
use crate::types::ops::BinaryOps;

#[derive(Clone, Eq, PartialEq)]
pub struct Tensor<T: TensorTrait<T>> {
    pub lazy_data: LazyBuffer<T>,
    requires_grad: bool,
    pub op: Ops,
    // array of parents
    pub prev: Option<Box<Vec<Tensor<T>>>>,
    pub gradient: Option<Box<Tensor<T>>>,
    pub unique_id: i32,
}

// implement hash trait for tensor struct
impl<T> Hash for Tensor<T> where T: TensorTrait<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.lazy_data.hash(state);
        self.requires_grad.hash(state);
        self.op.hash(state);
        self.prev.hash(state);
        self.gradient.hash(state);
    }
}

impl<T> Tensor<T> where T: TensorTrait<T> {
    /// Create a new tensor.
    ///
    /// # Arguments
    ///
    /// * `data` - The data of the tensor. Stored as a boxed slice of type T.
    /// * `dimensions` - The dimensions of the tensor.
    /// * `device` - The device to store the tensor on. Currently unused.
    /// * `requires_grad` - Whether or not the tensor requires gradients. This should be true if tensor is involved in neural network.
    ///
    /// # Panics
    ///
    /// * If data length does not match dimensions.
    /// * If unable to convert random number to i32 when generating unique id.
    pub fn new(
        data: DataArray<T>,
        dimensions: Dimensions,
        device: Option<Device>,
        requires_grad: Option<bool>
    ) -> Self {
        if data.len() != dimensions.0 * dimensions.1 {
            panic!("Data length does not match dimensions");
        }
        let requires_grad = match requires_grad {
            Some(requires_grad) => requires_grad,
            None => false,
        };
        let lazy_data: LazyBuffer<T> = LazyBuffer::new(data, dimensions, device);
        let new_op = Ops::None;
        // create unique id
        let rand_id = random_number(T::zero(), T::one()).to_f32();
        let rand_id: i32 = match rand_id {
            Some(rand_id) => (rand_id * 10000000.0) as i32,
            None => panic!("Error converting random number to i32"),
        };

        // create gradient placeholder array
        Self {
            lazy_data,
            requires_grad,
            op: new_op,
            prev: None,
            gradient: requires_grad.then(|| Box::from(Tensor::zeros(dimensions, None, None))),
            unique_id: rand_id,
        }
    }

    /// Create a new tensor with full control over all fields. This is typically only needed for internal use.
    ///
    /// # Arguments
    ///
    /// * `data` - The data of the tensor. Stored as a boxed slice of type T.
    /// * `dimensions` - The dimensions of the tensor.
    /// * `device` - The device to store the tensor on. Currently unused.
    /// * `requires_grad` - Whether or not the tensor requires gradients. This should be true if tensor is involved in neural network.
    /// * `op` - The operation that created this tensor.
    /// * `prev` - The previous tensors that were used to create this tensor.
    pub fn _build_raw(
        data: DataArray<T>,
        dimensions: Dimensions,
        device: Option<Device>,
        requires_grad: Option<bool>,
        op: Option<Ops>,
        prev: Option<Vec<Tensor<T>>>
    ) -> Tensor<T> {
        Self::new_internal(data, dimensions, device, requires_grad, op, prev)
    }

    fn new_internal(
        data: DataArray<T>,
        dimensions: Dimensions,
        device: Option<Device>,
        requires_grad: Option<bool>,
        op: Option<Ops>,
        prev: Option<Vec<Tensor<T>>>
    ) -> Self {
        if data.len() != dimensions.0 * dimensions.1 {
            panic!("Data length does not match dimensions");
        }
        let requires_grad = match requires_grad {
            Some(requires_grad) => requires_grad,
            None => false,
        };
        let lazy_data: LazyBuffer<T> = LazyBuffer::new(data, dimensions, device);
        let new_op = match op {
            Some(op) => op,
            None => Ops::None,
        };
        let new_prev = match prev {
            Some(prev) => prev,
            None => vec![],
        };
        let rand_id = random_number(T::zero(), T::one()).to_f32();
        // create unique id
        let rand_id: i32 = match rand_id {
            Some(rand_id) => (rand_id * 10000000.0) as i32,
            None => panic!("Error converting random number to i32"),
        };
        Self {
            lazy_data,
            requires_grad,
            op: new_op,
            prev: Some(Box::new(new_prev)),
            gradient: None,
            unique_id: rand_id,
        }
    }

    /// Get dimensions of tensor
    ///
    /// # Returns
    ///
    /// * `dimensions` - The dimensions of the tensor.
    ///
    pub fn dim(&self) -> Dimensions {
        self.lazy_data.dim()
    }
    /// Get data of tensor
    ///
    /// # Returns
    ///
    /// * `data` - The data of the tensor. Stored as a boxed slice of type T.
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

    /// Exchange rows and columns of tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice();
    /// let mut tensor = Tensor::new(data, (2, 2), None, None);
    /// tensor.transpose();
    ///
    /// assert_eq!(tensor.data(), &vec![1.0, 3.0, 2.0, 4.0].into_boxed_slice());
    /// ```
    pub fn transpose(&mut self) {
        let dim: Dimensions = self.dim();
        let mut new_data = Vec::with_capacity(dim.0 * dim.1);
        let data: &DataArray<T> = self.data();
        for i in 0..dim.1 {
            for j in 0..dim.0 {
                let index = j * dim.1 + i;
                new_data.push(data[index]);
            }
        }
        let new_data: DataArray<T> = new_data.into_boxed_slice();
        self.lazy_data = LazyBuffer::new(new_data, (dim.1, dim.0), None);
    }

    /// Compute sum of all elements in tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice();
    ///
    /// let tensor = Tensor::new(data, (2, 2), None, None);
    /// let sum = tensor.sum();
    ///
    /// assert_eq!(sum, 10.0);
    /// ```
    ///
    /// # Returns
    ///
    /// * `sum` - The sum of all elements in tensor.
    pub fn sum(&self) -> T {
        let dim: Dimensions = self.dim();
        let mut sum = T::zero();
        let data: &DataArray<T> = self.data();
        for i in 0..dim.0 {
            for j in 0..dim.1 {
                let index = i * dim.1 + j;
                sum = sum + data[index];
            }
        }
        sum
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

    //
    // Generate a tensor with random values drawn from a uniform distribution between 0 and 1.
    //
    // # Arguments
    // * `dim` - The dimensions of the tensor.
    // * `device` - The device to store the tensor on.
    // * `requires_grad` - Whether or not the tensor requires gradients.
    pub fn rand(dim: Dimensions, device: Option<Device>, requires_grad: Option<bool>) -> Self {
        let mut new_data = Vec::with_capacity(dim.0 * dim.1);
        let mut i: usize = 0;
        while i < dim.0 * dim.1 {
            new_data.push(random_number(T::zero(), T::one()));
            i += 1;
        }
        Self::new(new_data.into_boxed_slice(), dim, device, requires_grad)
    }
    ///
    /// Generate a tensor with random values from a uniform distribution.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimensions of the tensor.
    /// * `low` - The lower bound of the uniform distribution.
    /// * `high` - The upper bound of the uniform distribution.
    /// * `device` - The device to store the tensor on.
    /// * `requires_grad` - Whether or not the tensor requires gradients.
    pub fn uniform(
        dim: Dimensions,
        low: T,
        high: T,
        device: Option<Device>,
        requires_grad: Option<bool>
    ) -> Self {
        let mut new_data = Vec::with_capacity(dim.0 * dim.1);
        let mut i: usize = 0;
        while i < dim.0 * dim.1 {
            new_data.push(random_number(low, high));
            i += 1;
        }
        Self::new(new_data.into_boxed_slice(), dim, device, requires_grad)
    }

    /// Set gradient of tensor
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient to set.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::Tensor;
    ///
    /// let mut tensor:Tensor<f64> = Tensor::ones((2, 2), None, None);
    /// tensor.set_gradient(Tensor::zeros((2, 2), None, None));
    /// ```
    ///
    pub fn set_gradient(&mut self, gradient: Tensor<T>) {
        self.gradient = Some(Box::from(gradient));
    }

    /// Get gradient of tensor
    ///
    /// # Returns
    ///
    /// * `gradient` - The gradient of the tensor. May be None.
    pub fn get_gradient(&self) -> &Option<Box<Tensor<T>>> {
        &self.gradient
    }

    /// Compute backward pass of tensor and its parents. This will update each parent's gradient.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::Tensor;
    ///
    /// let mut tensor:Tensor<f64> = Tensor::ones((2, 2), None, None);
    /// tensor.backward();
    /// ```
    ///
    pub fn backward(&mut self) {
        // check if requires grad
        if !self.requires_grad {
            return;
        }
        let mut new_visited: Vec<i32> = Vec::new();
        // fill gradient with ones
        self.set_gradient(Tensor::ones(self.dim(), None, None));
        // print gradient of tensor
        println!("gradient of tensor: {:}", self.get_gradient().as_ref().unwrap());
        // run backward pass
        backward_helper(&mut new_visited, self)
    }

    pub fn max(&self, other: T) -> Tensor<T> {
        let dim: Dimensions = self.dim();
        let mut new_data = Vec::with_capacity(dim.0 * dim.1);
        let data: &DataArray<T> = self.data();
        for i in 0..dim.0 * dim.1 {
            new_data.push(if data[i] > other { data[i] } else { other });
        }
        let new_data: DataArray<T> = new_data.into_boxed_slice();
        let mut new_tensor = Tensor::_build_raw(
            new_data,
            dim,
            None,
            Some(true),
            Some(Ops::BinaryOps(BinaryOps::MAX)),
            Some(vec![self.clone()])
        );
        new_tensor.set_gradient(Tensor::zeros(dim, None, None));
        new_tensor
    }
}

// TODO: ONLY ADD GRADIENT/PREV IF REQUIRES GRAD IS TRUE
// math helpers
fn add<T: TensorTrait<T>>(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
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
    let mut new_tensor = Tensor::new_internal(
        new_data,
        a_dim,
        None,
        Some(true),
        Some(Ops::BinaryOps(BinaryOps::ADD)),
        Some(vec![a, b])
    );
    new_tensor.set_gradient(Tensor::zeros(a_dim, None, None));
    new_tensor
}

fn mul<T: TensorTrait<T>>(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
    // make sure dimensions match
    let a_dim: Dimensions = a.dim();
    let b_dim: Dimensions = b.dim();
    if !is_valid_matrix_multiplication(a_dim, b_dim) {
        panic!("Invalid matrix multiplication");
    }
    let mut new_data = Vec::with_capacity(a_dim.0 * b_dim.1);
    // print new data len
    let a_data = a.data();
    let b_data = b.data();
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
    // create Box<[T]> from Vec<T>
    let new_data: DataArray<T> = new_data.into_boxed_slice();
    let new_dim: Dimensions = new_dimensions_after_matrix_multiplication(a_dim, b_dim);
    let mut new_tensor = Tensor::new_internal(
        new_data,
        new_dim,
        None,
        Some(true),
        Some(Ops::BinaryOps(BinaryOps::MUL)),
        Some(vec![a, b])
    );
    new_tensor.set_gradient(Tensor::zeros(new_dim, None, None));
    new_tensor
}

fn sub<T: TensorTrait<T>>(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
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
    let mut new_tensor = Tensor::new_internal(
        new_data,
        a_dim,
        None,
        Some(true),
        Some(Ops::BinaryOps(BinaryOps::SUB)),
        Some(vec![a, b])
    );
    new_tensor.set_gradient(Tensor::zeros(a_dim, None, None));
    new_tensor
}

// addition
impl<T> Add<Tensor<T>> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn add(self, other: Tensor<T>) -> Tensor<T> {
        add(self, other)
    }
}

// subtraction
impl<T> Sub<Tensor<T>> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn sub(self, other: Tensor<T>) -> Tensor<T> {
        sub(self, other)
    }
}

// multiplication
impl<T> Mul<Tensor<T>> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        mul(self, other)
    }
}

// TODO: CHECK WHAT NODES SHOULD BE ADDED AS PARENTS AND WHAT GRADIENT SHOULD BE
// multiplication by scalar
impl<T> Mul<T> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn mul(self, other: T) -> Tensor<T> {
        // panic for now while we figure out how to implement this
        panic!("Not implemented");
        let dim: Dimensions = self.dim();
        let mut new_data = Vec::with_capacity(dim.0 * dim.1);
        let data: &DataArray<T> = self.data();
        for i in 0..dim.0 * dim.1 {
            new_data.push(data[i] * other);
        }
        let new_data: DataArray<T> = new_data.into_boxed_slice();
        let mut new_tensor = Tensor::_build_raw(
            new_data,
            dim,
            None,
            Some(true),
            Some(Ops::BinaryOps(BinaryOps::MUL)),
            Some(vec![self])
        );
        new_tensor.set_gradient(Tensor::zeros(dim, None, None));
        new_tensor
    }
}

// implement display trait for lazy buffer
impl<T> fmt::Display for Tensor<T> where T: TensorTrait<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // diplay array as matrix
        let dim: Dimensions = self.dim();
        let mut i: usize = 0;
        println!("\n");
        write!(f, "tensor( ")?;
        while i < dim.0 {
            if i != 0 {
                write!(f, "\t")?;
            }
            let mut j: usize = 0;
            while j < dim.1 {
                let index = i * dim.0 + j;
                write!(f, "{:} ", self.data()[index])?;
                j += 1;
            }
            i += 1;
            if i < dim.0 {
                write!(f, "\n")?;
            }
        }
        write!(f, ")\n")
    }
}
