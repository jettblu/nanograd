use crate::{ Tensor, TensorTrait };

pub struct Linear<T: TensorTrait<T>> {
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
}

impl<T: TensorTrait<T>> Linear<T> {
    pub fn new(in_features: usize, out_features: usize, bias: Option<bool>) -> Self {
        let new_dim = (in_features, out_features);
        let new_weight = Tensor::rand(new_dim, None, Some(true));
        let new_bias: Option<Tensor<T>> = match bias {
            Some(b) => {
                if b { Some(Tensor::zeros(new_dim, None, Some(true))) } else { None }
            }
            None => None,
        };
        Linear {
            weight: new_weight,
            bias: new_bias,
        }
    }
    pub fn forward(mut self, input: Tensor<T>) -> Tensor<T> {
        // transpose the weight matrix
        self.weight.transpose();
        let output = input * self.weight;
        match self.bias {
            Some(b) => output + b,
            None => output,
        }
    }
}
