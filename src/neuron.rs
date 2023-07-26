use crate::value::Value;
use rand::{ thread_rng, Rng };

// struct adopted from https://github.com/danielway/micrograd-rs
#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    /// Create a new neuron with random weights and bias.
    ///
    /// # Arguments
    ///
    /// * `input_count` - The number of inputs to the neuron.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::neuron::Neuron;
    ///
    /// let neuron = Neuron::new(2);
    /// ```
    pub fn new(input_count: usize) -> Neuron {
        let mut rng = thread_rng();
        let mut rand_value_fn = || {
            let data = rng.gen_range(-1.0..1.0);
            Value::from(data)
        };

        let mut weights = Vec::new();
        for _ in 0..input_count {
            weights.push(rand_value_fn());
        }

        Neuron {
            weights,
            bias: rand_value_fn().with_label("b"),
        }
    }

    /// Run a forward pass on the neuron.
    ///
    /// # Arguments
    ///
    /// * `xs` - The inputs to the neuron.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::neuron::Neuron;
    /// use nanograd::value::Value;
    ///
    /// let neuron = Neuron::new(2);
    /// let a = Value::from(1.0);
    /// let b = Value::from(2.0);
    /// let c = neuron.forward(&vec![a, b]);
    /// ```
    ///
    pub fn forward(&self, xs: &Vec<Value>) -> Value {
        // zip weights and inputs, multiply, and collect into a vector
        let products = std::iter
            ::zip(&self.weights, xs)
            .map(|(a, b)| a * b)
            .collect::<Vec<Value>>();

        // sum the products (of each weight, input pair) and add the bias
        // TODO: make it possible to specify different sum constant other than bias
        let sum =
            self.bias.clone() +
            products
                .into_iter()
                .reduce(|acc, prd| acc + prd)
                .unwrap();
        sum.tanh()
    }

    /// Get the weights and bias of the neuron.
    ///
    /// # Examples
    ///
    /// ```
    /// use nanograd::neuron::Neuron;
    ///
    /// let neuron = Neuron::new(2);
    /// let params = neuron.parameters();
    /// ```
    ///
    pub fn parameters(&self) -> Vec<Value> {
        [self.bias.clone()].into_iter().chain(self.weights.clone()).collect()
    }
}
