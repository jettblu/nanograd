use crate::{ neuron::Neuron, value::Value };

// struct adopted from https://github.com/danielway/micrograd-rs
#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Create a new layer of neurons
    ///
    /// # Arguments
    ///
    /// * `input_count` - The number of inputs to each neuron in the layer.
    /// * `output_count` - The number of neurons in the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::Layer;
    ///
    /// let layer = Layer::new(2, 1);
    /// ```
    ///
    pub fn new(input_count: usize, output_count: usize) -> Layer {
        Layer {
            neurons: (0..output_count).map(|_| Neuron::new(input_count)).collect(),
        }
    }

    /// Run a forward pass on the layer.
    ///
    /// # Arguments
    ///
    /// * `xs` - The inputs to the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::Layer;
    /// use nanograd::Value;
    ///
    /// let layer = Layer::new(2, 1);
    /// let a = Value::from(1.0);
    /// let b = Value::from(2.0);
    /// let c = layer.forward(&vec![a, b]);
    /// ```
    ///
    pub fn forward(&self, xs: &Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|n| n.forward(xs))
            .collect()
    }

    /// Get the weights and bias for each neuron within the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::Layer;
    ///
    /// let layer = Layer::new(2, 1);
    /// let params = layer.parameters();
    /// ```
    ///
    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .flat_map(|n| n.parameters())
            .collect()
    }
}
