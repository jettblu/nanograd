use getrandom::getrandom;

struct Tensor {
    data: Vec<f32>,
    device: Option<string>,
}

impl Tensor {
    // new tensor
    fn new(data: Vec<f32>) -> Self {
        Self { data, device: None }
    }
    // uniform random tensor
    fn randn(shape: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..shape.iter().product()).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { data, device: None }
    }

    // random tensor
    fn rand(shape: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..shape.iter().product()).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { data, device: None }
    }

    fn full(shape: Vec<usize>, fill_value: f32) -> Self {
        let data = (0..shape.iter().product()).map(|_| fill_value).collect();
        Self { data, device: None }
    }
}
