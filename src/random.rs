use getrandom::getrandom;

fn randomNumber(low: u8, high: u8) -> f64 {
    let mut buffer = [0u8; 4]; // A buffer to hold the random bytes

    // Generate random bytes using getrandom
    if let Err(err) = getrandom(&mut buffer) {
        eprintln!("Error generating random bytes: {}", err);
        return Value::from(0.0);
    }

    // Convert the bytes into a u32 random number
    let random_number = u32::from_ne_bytes(buffer);

    // Convert the u32 random number to a floating-point number between 0 and 1
    let random_float = (random_number as f64) / (u32::MAX as f64);

    // Map the range [0, 1] to the range [low, high]
    let constrained_random = (low as f64) + random_float * ((high as f64) - (low as f64));
    return constrained_random;
}
