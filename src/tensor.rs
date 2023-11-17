use core::panic;
use std::borrow::BorrowMut;
use std::fmt;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
use std::vec;

use crate::DataArray;
use crate::Device;
use crate::Dimensions;
use crate::LazyBuffer;
use crate::Ops;
use crate::TensorTrait;
use crate::backward::orchestrator::backward_by_operation;
use crate::helpers::is_valid_matrix_multiplication;
use crate::helpers::new_dimensions_after_matrix_multiplication;
use crate::random::random_number;
use crate::types::ops::BinaryOps;

#[derive(Clone, Eq, PartialEq)]
pub struct Tensor<T: TensorTrait<T>> {
    pub lazy_data: LazyBuffer<T>,
    requires_grad: bool,
    pub op: Ops,
    pub left: Option<TensorRef<T>>,
    pub right: Option<TensorRef<T>>,
    pub gradient: Option<TensorRef<T>>,
    pub unique_id: i32,
}

pub type TensorRef<T> = Box<Tensor<T>>;

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
            left: None,
            right: None,
            gradient: requires_grad.then(|| Box::from(Tensor::zeros(dimensions, None, None))),
            unique_id: rand_id,
        }
    }

    pub fn from_vec(
        data: Vec<T>,
        dimensions: Dimensions,
        device: Option<Device>,
        requires_grad: Option<bool>
    ) -> Self {
        Self::new(data.into_boxed_slice(), dimensions, device, requires_grad)
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
        left: Option<Tensor<T>>,
        right: Option<Tensor<T>>
    ) -> Tensor<T> {
        Self::new_internal(data, dimensions, device, requires_grad, op, left, right)
    }

    fn new_internal(
        data: DataArray<T>,
        dimensions: Dimensions,
        device: Option<Device>,
        requires_grad: Option<bool>,
        op: Option<Ops>,
        left: Option<Tensor<T>>,
        right: Option<Tensor<T>>
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
        let rand_id = random_number(T::zero(), T::one()).to_f32();
        // create unique id
        let rand_id: i32 = match rand_id {
            Some(rand_id) => (rand_id * 10000000.0) as i32,
            None => panic!("Error converting random number to i32"),
        };
        let new_left = match left {
            None => None,
            Some(left) => { Some(Box::from(left)) }
        };
        let new_right = match right {
            None => None,
            Some(right) => { Some(Box::from(right)) }
        };
        Self {
            lazy_data,
            requires_grad,
            op: new_op,
            left: new_left,
            right: new_right,
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

    pub fn set_op(&mut self, op: Ops) {
        self.op = op;
    }

    pub fn set_dim(&mut self, new_dim: Dimensions) {
        self.lazy_data.set_dim(new_dim);
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
    pub fn get_gradient(&self) -> Option<&TensorRef<T>> {
        match &self.gradient {
            None => None,
            Some(gradient) => Some(gradient),
        }
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
    pub fn backward(&mut self, grad: Option<&TensorRef<T>>, sibling: Option<&TensorRef<T>>) {
        let op = self.op;
        // ASSUMING THE ROOT TENSOR IS THE ONLY TENSOR WITH NO OP
        if op == Ops::None && grad.is_none() {
            // fill gradient with ones
            self.set_gradient(Tensor::ones(self.dim(), None, None));
        }
        if op != Ops::None && grad.is_none() && self.gradient.is_none() {
            panic!("Gradient is none for non root tensor");
        }
        let grad_to_pass: &Box<Tensor<T>> = self.gradient.as_ref().unwrap();
        if self.left.is_some() {
            let sibling: Option<&Box<Tensor<T>>> = match &self.right {
                None => None,
                Some(right) => Some(right),
            };
            self.left.as_mut().unwrap().backward(Some(grad_to_pass), sibling);
        }
        if self.right.is_some() {
            let sibling: Option<&Box<Tensor<T>>> = match &self.left {
                None => None,
                Some(left) => Some(left),
            };
            self.right.as_mut().unwrap().backward(Some(grad_to_pass), sibling);
        }
    }

    ///
    /// Fill diagonal of tensor with value
    ///
    /// # Arguments
    ///
    /// * `value` - The value to fill the diagonal with.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::Tensor;
    ///
    /// let mut tensor:Tensor<f64> = Tensor::ones((2, 2), None, None);
    /// tensor.fill_diagonal(2.0);
    /// ```
    ///
    pub fn fill_diagonal(&mut self, value: T) {
        let dim: Dimensions = self.dim();
        let mut i: usize = 0;
        let mut j: usize = 0;
        let mut new_data = vec![T::zero(); dim.0 * dim.1];
        let mut index: usize = 0;
        while i < dim.0 && j < dim.1 {
            index = i * dim.1 + j;
            new_data[index] = value;
            i += 1;
            j += 1;
        }
        let new_data: DataArray<T> = new_data.into_boxed_slice();
        self.lazy_data = LazyBuffer::new(new_data, dim, None);
    }

    pub fn flatten(&mut self) {
        let dim: Dimensions = self.dim();
        self.set_dim((1, dim.0 * dim.1));
    }
}

// TODO: ONLY ADD GRADIENT/PREV IF REQUIRES GRAD IS TRUE
// math helpers
fn add<T: TensorTrait<T>>(mut a: Tensor<T>, mut b: Tensor<T>) -> Tensor<T> {
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
    a.set_op(Ops::BinaryOps(BinaryOps::ADD));
    b.set_op(Ops::BinaryOps(BinaryOps::ADD));
    let mut new_tensor = Tensor::new_internal(
        new_data,
        a_dim,
        None,
        Some(true),
        Some(Ops::None),
        Some(a),
        Some(b)
    );
    new_tensor.set_gradient(Tensor::zeros(a_dim, None, None));
    new_tensor
}

fn mul<T: TensorTrait<T>>(mut a: Tensor<T>, mut b: Tensor<T>) -> Tensor<T> {
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
    a.set_op(Ops::BinaryOps(BinaryOps::MUL));
    b.set_op(Ops::BinaryOps(BinaryOps::MUL));
    let mut new_tensor = Tensor::new_internal(
        new_data,
        new_dim,
        None,
        Some(true),
        Some(Ops::None),
        Some(a),
        Some(b)
    );
    new_tensor.set_gradient(Tensor::zeros(new_dim, None, None));
    new_tensor
}

fn sub<T: TensorTrait<T>>(mut a: Tensor<T>, mut b: Tensor<T>) -> Tensor<T> {
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
    a.set_op(Ops::BinaryOps(BinaryOps::SUB));
    b.set_op(Ops::BinaryOps(BinaryOps::SUB));
    let mut new_tensor = Tensor::new_internal(
        new_data,
        a_dim,
        None,
        Some(true),
        Some(Ops::None),
        Some(a),
        Some(b)
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

// multiplication by scalar
impl<T> Mul<T> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn mul(self, other: T) -> Tensor<T> {
        // create new diagonal matrix with values of other
        let dim: Dimensions = self.dim();
        let mut new_constant_tensor = Tensor::zeros(dim, None, Some(true));
        new_constant_tensor.fill_diagonal(other);
        mul(self, new_constant_tensor)
    }
}

// addition by scalar
impl<T> Add<T> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn add(self, other: T) -> Tensor<T> {
        // create new diagonal matrix with values of other
        let dim: Dimensions = self.dim();
        let new_constant_tensor = Tensor::full(dim, other, None, Some(true));
        add(self, new_constant_tensor)
    }
}

// subtraction by scalar
impl<T> Sub<T> for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn sub(self, other: T) -> Tensor<T> {
        // create new diagonal matrix with values of other
        let dim: Dimensions = self.dim();
        let mut new_constant_tensor = Tensor::full(dim, other, None, Some(true));
        sub(self, new_constant_tensor)
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

// implement negation trait for tensor

impl<T> Neg for Tensor<T> where T: TensorTrait<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> {
        // multiply by matrix with -1 on diagonal
        let dim: Dimensions = self.dim();
        let mut new_constant_tensor = Tensor::zeros(dim, None, Some(true));
        new_constant_tensor.fill_diagonal(T::zero() - T::one());
        mul(self, new_constant_tensor)
    }
}
