use getrandom::getrandom;

use crate::TensorTrait;

/// Generate a random number between low and high.
///
/// # Arguments
///
/// * `low` - The lower bound of the random number.
/// * `high` - The upper bound of the random number.
///
/// # Examples
///
/// ```
/// use nanograd::random_number;
///
/// let random_number:f64 = random_number(0.0, 1.0);
/// ```
///
pub fn random_number<T: TensorTrait<T>>(low: T, high: T) -> T {
    let mut buffer = [0u8; 4]; // A buffer to hold the random bytes

    // Generate random bytes using getrandom
    if let Err(err) = getrandom(&mut buffer) {
        panic!("Error generating random bytes: {}", err);
    }

    // Convert the bytes into a u32 random number
    let random_unsigned = u32::from_le_bytes(buffer);

    // Convert the u32 random number to a floating-point number between 0 and 1
    let random_float = (random_unsigned as f64) / (u32::MAX as f64);

    let result_num = T::try_from(random_float);
    if result_num.is_err() {
        panic!("Error converting random float to tensor type.");
    }
    let random_num: T;
    match result_num {
        Ok(res) => {
            random_num = res;
        }
        #[allow(unused_variables)]
        Err(err) => panic!("Error converting random float to tensor type"),
    }
    // Map the range [0, 1] to the range [low, high]
    let constrained_random = low + random_num * (high - low);
    return constrained_random;
}
