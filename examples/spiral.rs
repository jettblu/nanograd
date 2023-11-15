mod datasets;
mod sample;

use std::borrow::BorrowMut;

use nanograd::{ FeaturesAndLabels, nn::{ linear::Linear, activation::tanh }, TensorTrait, Tensor };

use crate::datasets::shapes::{ fetch_shape_dataset, ShapesDataset };
use crate::sample::random_unique_numbers;

struct TinyNet<T: TensorTrait<T>> {
    l1: Linear<T>,
    l2: Linear<T>,
}

impl<T: TensorTrait<T>> TinyNet<T> {
    fn new() -> Self {
        TinyNet {
            l1: Linear::new(2, 5, None),
            l2: Linear::new(5, 2, None),
        }
    }
    fn forward(&mut self, x: Tensor<T>) -> Tensor<T> {
        let self_l1 = self.l1.borrow_mut();
        let self_l2 = &mut self.l2;
        let x_1 = self.l1.forward(x);
        let x_2 = self.l2.forward(x_1);
        let x_3 = tanh(x_2);
        x_3
    }
}

// run this example with:
// cargo run --example spiral
fn main() {
    println!("Beginning spiral example...");
    let dataset: ShapesDataset = fetch_shape_dataset("spiral").unwrap();
    let mut net: TinyNet<f32> = TinyNet::new();
    // run through 1000 iterations
    for _ in 0..1000 {
        // we want to draw a random sample of 32 from the dataset
        let sample_size = 32;
        // the train dataset has 69 values
        let max_index = 69;
        let sample_indices = random_unique_numbers(max_index, sample_size);
        // take a random sample from the dataset... should be 32 samples
        let train_features = sample_indices
            .iter()
            .map(|i| dataset.train_features[*i])
            .collect::<Vec<f32>>();
        let train_labels = sample_indices
            .iter()
            .map(|i| dataset.train_labels[*i])
            .collect::<Vec<f32>>();
        // create a tensor from the train features
        let train_features = Tensor::new(
            train_features.into_boxed_slice(),
            (sample_size, 2),
            None,
            Some(true)
        );
        // create a tensor from the train labels
        let train_labels = Tensor::new(
            train_labels.into_boxed_slice(),
            (sample_size, 2),
            None,
            Some(true)
        );
        let y = net.forward(train_features);
    }
}
