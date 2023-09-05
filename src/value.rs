use std::{
    cell::{ Ref, RefCell },
    collections::HashSet,
    fmt::Debug,
    hash::Hash,
    iter::Sum,
    ops::{ Add, Deref, Mul, Neg, Sub },
    rc::Rc,
};

#[derive(Copy, Clone)]
pub enum Operation {
    Add,
    Mul,
    Tanh,
    Exp,
    None,
}

// implement hash trait for operation enum
impl Hash for Operation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Operation::Add => (0).hash(state),
            Operation::Mul => (1).hash(state),
            Operation::Tanh => (2).hash(state),
            Operation::Exp => (3).hash(state),
            Operation::None => (4).hash(state),
        }
    }
}

// test whether two operation enum vals are equal
impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Operation::Add, Operation::Add) => true,
            (Operation::Mul, Operation::Mul) => true,
            (Operation::Tanh, Operation::Tanh) => true,
            (Operation::None, Operation::None) => true,
            _ => false,
        }
    }
}

impl Debug for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Add => write!(f, "Add"),
            Operation::Mul => write!(f, "Mul"),
            Operation::Tanh => write!(f, "Tanh"),
            Operation::Exp => write!(f, "Exp"),
            Operation::None => write!(f, "None"),
        }
    }
}

// wrapped struct code adopted from https://github.com/danielway/micrograd-rs
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Value(Rc<RefCell<ValueInternal>>);

impl Value {
    pub fn from<T>(t: T) -> Value where T: Into<Value> {
        t.into()
    }

    fn new(value: ValueInternal) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    pub fn with_label(self, label: &str) -> Value {
        self.borrow_mut().label = Some(label.to_string());
        self
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn operation(&self) -> Operation {
        self.borrow().operation
    }

    pub fn gradient(&self) -> f64 {
        self.borrow().gradient
    }

    pub fn clear_gradient(&self) {
        self.borrow_mut().gradient = 0.0;
    }

    pub fn adjust(&self, factor: f64) {
        let mut value = self.borrow_mut();
        value.data += factor * value.gradient;
    }

    pub fn pow(&self, other: &Value) -> Value {
        let result = self.borrow().data.powf(other.borrow().data);

        Value::new(
            ValueInternal::new(result, None, Operation::Exp, vec![self.clone(), other.clone()])
        )
    }

    pub fn tanh(&self) -> Value {
        let result = self.borrow().data.tanh();

        Value::new(ValueInternal::new(result, None, Operation::Tanh, vec![self.clone()]))
    }

    pub fn backward(&self) {
        let mut visited: HashSet<Value> = HashSet::new();

        self.borrow_mut().gradient = 1.0;
        self.backward_internal(&mut visited, self);
    }

    fn backward_internal(&self, visited: &mut HashSet<Value>, value: &Value) {
        if !visited.contains(&value) {
            visited.insert(value.clone());

            let borrowed_value = value.borrow();
            if value.operation() != Operation::None {
                backward_by_operation(&borrowed_value);
            }

            for parent_id in &value.borrow().previous {
                self.backward_internal(visited, parent_id);
            }
        }
    }

    pub fn trace(&self) {
        let mut visited: HashSet<Value> = HashSet::new();
        println!("Tracing value...");
        println!();
        trace_internal(&mut visited, self);
        println!();
        println!("Done tracing value!")
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInternal>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(ValueInternal::new(t.into(), None, Operation::None, Vec::new()))
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(&self, &other)
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Self::Output {
        add(self, other)
    }
}

fn add(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data + b.borrow().data;

    Value::new(ValueInternal::new(result, None, Operation::Add, vec![a.clone(), b.clone()]))
}

// defines addition for f64 and Value
// in this case the f64 is on the right side of the addition
impl Add<f64> for Value {
    type Output = Value;

    fn add(self, other: f64) -> Self::Output {
        add(&self, &Value::from(other))
    }
}

// in this case the f64 is on the right side of the addition
impl Add<Value> for f64 {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(&Value::from(self), &other)
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        add(&self, &-other)
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        mul(&self, &other)
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Self::Output {
        mul(self, other)
    }
}

// defines multiplication for f64 and Value
// in this case the f64 is on the right side of the multiplication
impl Mul<f64> for Value {
    type Output = Value;

    fn mul(self, other: f64) -> Self::Output {
        mul(&self, &Value::from(other))
    }
}
// in this case the f64 is on the right side of the multiplication
impl Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        mul(&Value::from(self), &other)
    }
}

fn mul(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data * b.borrow().data;

    Value::new(ValueInternal::new(result, None, Operation::Mul, vec![a.clone(), b.clone()]))
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul(&self, &Value::from(-1))
    }
}

impl<'a> Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul(self, &Value::from(-1))
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut sum = Value::from(0.0);
        loop {
            let val = iter.next();
            if val.is_none() {
                break;
            }

            sum = sum + val.unwrap();
        }
        sum
    }
}

///
/// Internal representation of a Value
///
/// # Fields
/// * `data` - The data of the value
/// * `gradient` - The gradient of the value
/// * `label` - The label of the value
/// * `operation` - The operation of the value
/// * `previous` - The previous values of the value
/// * `propagate` - The propagate function of the value
///
/// # Methods
/// * `new` - Returns a new ValueInternal
///
pub struct ValueInternal {
    data: f64,
    gradient: f64,
    label: Option<String>,
    operation: Operation,
    previous: Vec<Value>,
}

impl ValueInternal {
    fn new(data: f64, label: Option<String>, op: Operation, prev: Vec<Value>) -> ValueInternal {
        ValueInternal {
            data,
            gradient: 0.0,
            label,
            operation: op,
            previous: prev,
        }
    }
}

impl PartialEq for ValueInternal {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data &&
            self.gradient == other.gradient &&
            self.label == other.label &&
            self.operation == other.operation &&
            self.previous == other.previous
    }
}

impl Eq for ValueInternal {}

impl Hash for ValueInternal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.gradient.to_bits().hash(state);
        self.label.hash(state);
        self.operation.hash(state);
        self.previous.hash(state);
    }
}

impl Debug for ValueInternal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueInternal")
            .field("data", &self.data)
            .field("gradient", &self.gradient)
            .field("label", &self.label)
            .field("operation", &self.operation)
            .field("previous", &self.previous)
            .finish()
    }
}

///
/// Backward by operation
///
/// # Arguments
/// * `val` - The value to apply the backward method to
fn backward_by_operation(val: &Ref<ValueInternal>) {
    match val.operation {
        Operation::Tanh => {
            let mut previous = val.previous[0].borrow_mut();
            previous.gradient += (1.0 - val.data.powf(2.0)) * val.gradient;
        }
        Operation::Exp => {
            let mut base = val.previous[0].borrow_mut();
            let power = val.previous[1].borrow();
            base.gradient += power.data * base.data.powf(power.data - 1.0) * val.gradient;
        }
        // gradient flows through plus signs
        Operation::Add => {
            let mut previous = val.previous[0].borrow_mut();
            previous.gradient += 1.0 * val.gradient;
            let mut previous = val.previous[1].borrow_mut();
            previous.gradient += 1.0 * val.gradient;
        }
        Operation::Mul => {
            let mut first = val.previous[0].borrow_mut();
            let mut second = val.previous[1].borrow_mut();
            first.gradient += second.data * val.gradient;
            second.gradient += first.data * val.gradient;
        }
        Operation::None => {
            println!("No operation when running backward method.");
        }
    }
}

fn trace_internal(visited: &mut HashSet<Value>, value: &Value) {
    if !visited.contains(&value) {
        visited.insert(value.clone());

        let borrowed_value = value.borrow();
        println!("{:?}", borrowed_value);
        // separate values with a newline
        println!();
        for child_id in &value.borrow().previous {
            trace_internal(visited, child_id);
        }
    }
}
