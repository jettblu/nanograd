use nanograd::Value;

fn main() {
    println!();
    println!("Running addition example...");
    println!();
    // add two values
    let a = Value::from(1.0);
    let b = Value::from(2.0);
    let c = a + b;
    // trace the value
    c.trace();
    println!();
    println!("Finished addition example.");
    println!();
}
