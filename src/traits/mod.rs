use std::ops::{ Add, Mul, Sub, Div };
use std::fmt::Display;
use std::fmt::Debug;
use std::cmp::PartialOrd;

extern crate num;
use num::{ ToPrimitive, FromPrimitive };
use self::num::pow::Pow;
use self::num::traits::Zero;
use self::num::traits::One;

pub trait TensorTrait<T>: Zero +
    One +
    Clone +
    Copy +
    TryFrom<f64> +
    Display +
    Debug +
    ToPrimitive +
    FromPrimitive +
    PartialOrd +
    Pow<T, Output = T> +
    Div<T, Output = T> +
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
            ToPrimitive +
            FromPrimitive +
            PartialOrd +
            Pow<T, Output = T> +
            Div<T, Output = T> +
            Add<T, Output = T> +
            Mul<T, Output = T> +
            Sub<T, Output = T> {}
