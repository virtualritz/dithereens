use dithereens::DitherParallelIteratorExt;
use rayon::prelude::*;

fn main() {
    let rng = wyrand::WyRand::new(42);

    // Generate some sample data
    let values: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();

    println!("Original values (first 10): {:?}", &values[..10]);

    // Use parallel iterator with simple_dither
    let dithered: Vec<f32> = values
        .par_iter()
        .copied()
        .map(|x| x * 0.8) // Scale down a bit
        .simple_dither(255.0, &rng);

    println!("Dithered values (first 10): {:?}", &dithered[..10]);

    // Use parallel iterator with full dither control
    let custom_dithered: Vec<f32> = values
        .par_iter()
        .copied()
        .map(|x| x * 0.8) // Scale down a bit
        .dither(0.0, 255.0, 0.5, &rng);

    println!(
        "Custom dithered values (first 10): {:?}",
        &custom_dithered[..10]
    );

    // Verify all values are in expected range
    assert!(dithered.iter().all(|&x| x >= 0.0 && x <= 255.0));
    println!("All simple_dither values are in range [0, 255]");

    // Custom dithered values might go slightly outside due to dithering
    println!(
        "Custom dithered range: [{}, {}]",
        custom_dithered.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        custom_dithered
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
}
