use crate::{ layer::Layer, value::Value };

// A multi-layer perceptron. This is a neural network with one or more hidden layers.
// Refer to this page for more information: https://en.wikipedia.org/wiki/Multilayer_perceptron

// struct adopted from https://github.com/danielway/micrograd-rs
#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// Create a new multi-layer perceptron.
    ///
    /// # Arguments
    ///
    /// * `input_count` - The number of inputs to the network.
    /// * `output_counts` - Number of nuerons in each layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::mlp::MLP;
    ///
    /// let mlp = MLP::new(2, vec![1]);
    /// ```
    ///
    pub fn new(input_count: usize, output_counts: Vec<usize>) -> MLP {
        let output_counts_len = output_counts.len();
        let layer_sizes: Vec<usize> = [input_count].into_iter().chain(output_counts).collect();

        MLP {
            layers: (0..output_counts_len)
                .map(|i| Layer::new(layer_sizes[i], layer_sizes[i + 1]))
                .collect(),
        }
    }

    /// Run a forward pass on the network.
    ///
    /// # Arguments
    ///
    /// * `xs` - The inputs to the network.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::mlp::MLP;
    ///
    /// let mlp = MLP::new(2, vec![1]);
    /// let a = vec![1.0, 2.0];
    /// let b = mlp.forward(a);
    /// ```
    ///
    pub fn forward(&self, mut xs: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            xs = layer.forward(&xs);
        }
        xs
    }

    /// Get the parameters of the network.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::mlp::MLP;
    ///
    /// let mlp = MLP::new(2, vec![1]);
    /// let params = mlp.parameters();
    /// ```
    ///
    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|l| l.parameters())
            .collect()
    }
}
