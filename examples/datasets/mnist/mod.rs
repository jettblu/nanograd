use byteorder::{ BigEndian, ReadBytesExt };
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{ Cursor, Read };

use crate::TensorTrait;
use crate::Tensor;

use nanograd::types::data::FeaturesAndLabels;

pub struct MnistImage {
    pub image: Tensor<f64>,
    pub classification: f64,
}

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

pub fn fetch_mnist<T: TensorTrait<T>>(
    dataset_name: &str
) -> Result<FeaturesAndLabels<T>, std::io::Error> {
    println!("{}", std::env::current_dir().unwrap().display());
    let base_path: &str = "src/extra/datasets/mnist/";
    let filename = format!("{}{}-labels-idx1-ubyte.gz", base_path, dataset_name);
    let label_data = &MnistData::new(&File::open(filename)?)?;
    let filename = format!("{}{}-images-idx3-ubyte.gz", base_path, dataset_name);
    let images_data = &MnistData::new(&File::open(filename)?)?;
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    let output: FeaturesAndLabels<T>;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data
            .into_iter()
            .map(|x| (x as f64) / 255.0)
            .collect();
        // print image data
        println!("{:?}", image_data);
        println!("image_data.len(): {:?}", image_data.len());
        print!("index: {:?}", i);
        println!("--------------------------");
        // let image = Tensor::new(image_data.into_boxed_slice(), (1, 784), None, Some(true));
    }
    Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "Not implemented"))
}
