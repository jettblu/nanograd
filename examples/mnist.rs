use nanograd::{ nn::{ linear::Linear, activation::tanh }, TensorTrait, Tensor };

struct TinyNet<T: TensorTrait<T>> {
    l1: Linear<T>,
    l2: Linear<T>,
}

impl<T: TensorTrait<T>> TinyNet<T> {
    fn new() -> Self {
        TinyNet {
            l1: Linear::new(784, 128, None),
            l2: Linear::new(128, 10, None),
        }
    }
    fn forward(self, x: Tensor<T>) -> Tensor<T> {
        let x_1 = self.l1.forward(x);
        let x_2 = self.l2.forward(x_1);
        let x_3 = tanh(x_2);
        x_3
    }
}

fn main() {
    panic!("This example is not yet implemented")
}
