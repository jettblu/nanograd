use crate::{ TensorTrait, Device, default_device };
use std::hash::Hash;

#[derive(Clone, PartialEq, Eq)]
pub struct LazyBuffer<T: TensorTrait<T>> {
    data: DataArray<T>,
    dimensions: Dimensions,
    device: Device,
    realized: bool,
    // prev
}

impl<T> LazyBuffer<T> where T: TensorTrait<T> {
    pub fn new(data: DataArray<T>, dimensions: Dimensions, device: Option<Device>) -> Self {
        let device = match device {
            Some(device) => device,
            None => default_device(),
        };
        Self { data, dimensions, device, realized: true }
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

    pub fn is_realized(&self) -> bool {
        self.realized
    }

    pub fn realize(&self) -> &LazyBuffer<T> {
        // TODO: implement
        // for now just return data since always realized
        self
    }
}

pub type DataArray<T> = Box<[T]>;

pub type Dimensions = (usize, usize);

impl<T> Hash for LazyBuffer<T> where T: TensorTrait<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {}
}
