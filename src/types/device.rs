#[derive(Clone, PartialEq, Eq)]
pub enum Device {
    CPU,
    OPENCL,
}

// TODO: add a function to detct env and set device accordingly
pub fn default_device() -> Device {
    Device::CPU
}
