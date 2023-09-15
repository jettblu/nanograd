use std::env;
use std::fs::File;
use std::io::BufReader;

use nanograd::{ FeaturesAndLabels, Tensor, Dimensions };
use serde::Deserialize;
use serde_json::{ from_reader };
// import error type
use serde_json::error::Error;

#[derive(Deserialize)]
struct Observation {
    features: Vec<f32>,
    label: f32,
}

pub struct ShapesDataset {
    pub train_features: Vec<f32>,
    pub test_features: Vec<f32>,
    pub train_labels: Vec<f32>,
    pub test_labels: Vec<f32>,
}

impl ShapesDataset {
    pub fn new(
        train_features: Vec<f32>,
        train_labels: Vec<f32>,
        test_features: Vec<f32>,
        test_labels: Vec<f32>
    ) -> Self {
        ShapesDataset {
            train_features,
            train_labels,
            test_features,
            test_labels,
        }
    }
}

// ignore unused warning
#[allow(dead_code)]
pub fn fetch_shape_dataset(dataset_name: &str) -> Result<ShapesDataset, std::io::Error> {
    let base_path: &str = "examples/datasets/shapes/";
    let filename = format!("{}{}.json", base_path, dataset_name);
    let file = File::open(&filename)?;
    let reader = BufReader::new(file);
    // load data set from json string
    let data_raw: Result<Vec<Observation>, Error> = from_reader(reader);
    let data: Vec<Observation>;
    match data_raw {
        Ok(d) => {
            // successfully loaded data set
            data = d;
        }
        Err(e) => {
            // error loading data set
            println!("Error: {:?}", e);
            panic!("Error loading spiral dataset!");
        }
    }
    let data_len = data.len();

    // convert data set to tensors
    let mut features: Vec<f32> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    for observation in data {
        features.extend(observation.features);
        labels.push(observation.label);
    }
    // create 70/30 train/test split
    let split_index = ((data_len as f32) * 0.7) as usize;
    let features_train = &features[0..split_index];
    let features_test = &features[split_index..];
    let labels_train = &labels[0..split_index];
    let labels_test = &labels[split_index..];

    let output: ShapesDataset = ShapesDataset::new(
        features_train.to_vec(),
        labels_train.to_vec(),
        features_test.to_vec(),
        labels_test.to_vec()
    );
    Ok(output)
}
