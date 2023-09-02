use std::ops::{ Add, Mul, Sub };
use std::fmt::Display;
use std::fmt::Debug;

extern crate num;
use self::num::traits::Zero;
use self::num::traits::One;
// use -1 trait
use self::num::traits::Signed;

pub trait TensorTrait<T>: Zero +
    One +
    Clone +
    Copy +
    TryFrom<f64> +
    Display +
    Debug +
    Add<T, Output = T> +
    Mul<T, Output = T> +
    Sub<T, Output = T> {}

impl<T> TensorTrait<T>
    for T
    where
        T: Zero +
            One +
            Clone +
            Copy +
            TryFrom<f64> +
            Display +
            Debug +
            Add<T, Output = T> +
            Mul<T, Output = T> +
            Sub<T, Output = T> {}
