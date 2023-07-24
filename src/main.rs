type Node = Option<Box<Value>>;
type Children = Option<(Node, Node)>;
use std::ops::Add;
use std::ops::Mul;
use std::collections::HashSet;

#[derive(Copy, Clone)]
enum Operation {
    Add,
    Mul,
    Tanh,
    None,
}

#[derive(Clone)]
struct Value {
    grad: f32,
    data: f32,
    label: String,
    prev: Children,
    op_type: Operation,
}

impl Value {
    ///
    /// Returns a new Value
    ///
    /// # Arguments
    /// * `data` - The data of the value
    /// * `label` - The label of the value
    fn new(data: f32, label: String) -> Value {
        Value {
            grad: 0.0,
            data: data,
            label: label,
            prev: None,
            op_type: Operation::None,
        }
    }

    fn tanh(&self) -> Value {
        Value {
            grad: 0.0,
            data: self.data.tanh(),
            label: String::from("tanh"),
            prev: Some((Some(Box::new(self.clone())), None)),
            op_type: Operation::Tanh,
        }
    }

    fn display(&self) {
        println!("Value {}, data: {}", self.label, self.data);
    }

    fn _backward(&mut self) {
        match self.op_type {
            Operation::Add => {
                let (ref mut _x1, ref mut _x2) = self.prev.as_mut().unwrap();
                let x1 = _x1.as_mut().unwrap();
                let x2 = _x2.as_mut().unwrap();
                x1.grad += 1.0 * self.grad;
                x2.grad += 1.0 * self.grad;
            }
            Operation::Mul => {
                // get other
                let (ref mut _x1, ref mut _x2) = self.prev.as_mut().unwrap();
                let x1 = _x1.as_mut().unwrap();
                let x2 = _x2.as_mut().unwrap();
                x1.grad += x2.data * self.grad;
                x2.grad += x1.data * self.grad;
            }
            Operation::Tanh => {
                let (ref mut _x1, _) = self.prev.as_mut().unwrap();
                let x1 = _x1.as_mut().unwrap();
                x1.grad = x1.data.atanh() * self.grad;
            }
            Operation::None => {
                println!("No operation");
            }
        }
    }

    fn backward(&mut self) {
        let topo = build_topo(self);
        self.grad = 1.0;
        for mut value in topo {
            value._backward();
        }
    }
}

// topological sort value
// must return a vec of topological sorted values
// do not call ._backward() on the values
fn build_topo(value: &mut Value) -> Vec<Value> {
    let mut topo: Vec<Value> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut stack: Vec<Value> = Vec::new();
    stack.push(value.clone());
    while stack.len() > 0 {
        let mut current = stack.pop().unwrap();
        if visited.contains(&current.label) {
            continue;
        }
        visited.insert(current.label.clone());
        match current.prev {
            Some((Some(ref mut x1), Some(ref mut x2))) => {
                stack.push(*x1.clone());
                stack.push(*x2.clone());
            }
            Some((Some(ref mut x1), None)) => {
                stack.push(*x1.clone());
            }
            _ => {}
        }
        topo.push(current);
    }
    topo.reverse();
    topo
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        Value {
            grad: 0.0,
            data: self.data + other.data,
            label: String::from("+"),
            prev: Some((Some(Box::new(self)), Some(Box::new(other)))),
            op_type: Operation::Add,
        }
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        Value {
            grad: 0.0,
            data: self.data * other.data,
            label: String::from("*"),
            prev: Some((Some(Box::new(self)), Some(Box::new(other)))),
            op_type: Operation::Mul,
        }
    }
}

fn main() {
    println!("Running example...");
    let x1 = Value::new(1.0, "test".to_string());
    let x2 = Value::new(2.0, "test 2".to_string());
    let x3 = x1 + x2;
    let x4 = x3.tanh();
    let mut x5 = x3 * x4;
    let res = build_topo(&mut x5);
    for value in res {
        value.display();
    }
    x5.backward();
    println!("DONE!");
}
