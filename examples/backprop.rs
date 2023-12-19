use nanograd::Tensor;

// to run this example:
// cargo run --example backprop
fn main() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![2.0, 2.0, 2.0, 2.0];
    let dim = (2, 2);
    let a = Tensor::from_vec(a_data.clone(), dim, None, Some(true));
    let b = Tensor::from_vec(b_data.clone(), dim, None, Some(true));
    let mut c = a * b;
    c.backward();
    c.print_path(1);
}
