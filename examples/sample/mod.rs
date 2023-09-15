use rand::Rng;

// this function should create an array of unique random numbers between 0 and n
// each random number should be an integer
// there should be no duplicates
// the array should be of length p
pub fn random_unique_numbers(maxVal: usize, arraySize: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut random_numbers: Vec<usize> = Vec::new();
    while random_numbers.len() < arraySize {
        let random_number = rng.gen_range(0..maxVal);
        if !random_numbers.contains(&random_number) {
            random_numbers.push(random_number);
        }
    }
    return random_numbers;
}
