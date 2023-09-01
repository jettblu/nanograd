use std::ops::{ Add, Mul };
use std::fmt::Display;
use std::fmt::Debug;

extern crate num;
use self::num::traits::Zero;
use self::num::traits::One;

pub trait TensorTrait<T>: Zero +
    One +
    Clone +
    Copy +
    Display +
    Debug +
    Add<T, Output = T> +
    Mul<T, Output = T> {}

impl<T> TensorTrait<T>
    for T
    where
        T: Zero + One + Clone + Copy + Display + Debug + Add<T, Output = T> + Mul<T, Output = T> {}
