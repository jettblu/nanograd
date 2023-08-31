use nanograd::{ Tensor };
use serde::Deserialize;

#[derive(Deserialize)]
struct Observation {
    features: Vec<f64>,
    label: f64,
}

fn main() {
    let data_a = vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice();
    let data_b = vec![10.0, 20.0, 30.0, 40.0].into_boxed_slice();
    let dimensions = (2, 2);
    let tensor_a = Tensor::new(data_a, dimensions, None, None);
    let tensor_b = Tensor::new(data_b, dimensions, None, None);
    let tensor_c = tensor_a + tensor_b;
    println!("{:?}", tensor_c.data());
}
