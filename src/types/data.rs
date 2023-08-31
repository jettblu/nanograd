use crate::{ TensorTrait, Device, default_device };

pub enum DType {
    float32,
    float64,
    int32,
    int64,
    uint32,
    uint64,
    bool,
}

pub fn default_data_type() -> DType {
    DType::float32
}

pub struct LazyBuffer<T: TensorTrait<T>> {
    data: DataArray<T>,
    dimensions: Dimensions,
    device: Device,
}

impl<T> LazyBuffer<T> where T: TensorTrait<T> {
    pub fn new(data: DataArray<T>, dimensions: Dimensions, device: Option<Device>) -> Self {
        let device = match device {
            Some(device) => device,
            None => default_device(),
        };
        Self { data, dimensions, device }
    }
    pub fn dim(&self) -> Dimensions {
        self.dimensions
    }
    pub fn data(&self) -> &DataArray<T> {
        &self.data
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
}

pub type DataArray<T> = Box<[T]>;

pub type Dimensions = (usize, usize);
