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
    let data_d = vec![5.0, 6.0, 7.0, 8.0].into_boxed_slice();
    let dim_a = (2, 2);
    let dim_b = (2, 2);
    let tensor_a = Tensor::new(data_a, dim_a, None, Some(true));
    let tensor_b = Tensor::new(data_b, dim_b, None, Some(true));
    let tensor_c = tensor_a * tensor_b;
    let tensor_d = Tensor::new(data_d, dim_a, None, Some(true));
    let mut tensor_e = tensor_c + tensor_d;
    tensor_e.backward();
    println!("{:}", tensor_e);
}
