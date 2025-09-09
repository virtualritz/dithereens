use dithereens::DitherParallelIteratorExt;
use rayon::prelude::*;

fn main() {
    // Generate some sample data
    let values: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();

    println!("Original values (first 10): {:?}", &values[..10]);

    // Use parallel iterator with simple_dither (uses Hash by default)
    let dithered: Vec<f32> = values
        .par_iter()
        .copied()
        .map(|x| x * 0.8) // Scale down a bit
        .simple_dither(255.0, 42);

    println!("Dithered values (first 10): {:?}", &dithered[..10]);

    // Use parallel iterator with full dither control
    let custom_dithered: Vec<f32> = values
        .par_iter()
        .copied()
        .map(|x| x * 0.8) // Scale down a bit
        .dither(0.0, 255.0, 0.5, 42);

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

    // Compare different methods in parallel
    println!("\n--- Comparing Different Methods in Parallel ---");

    let test_values = vec![0.5f32; 20];

    let hash = dithereens::Hash::new(42);
    let r2 = dithereens::R2::new(42);
    let golden = dithereens::GoldenRatio::new(42);

    let hash_result: Vec<f32> = test_values
        .par_iter()
        .copied()
        .simple_dither_with_method(255.0, &hash);

    let r2_result: Vec<f32> = test_values
        .par_iter()
        .copied()
        .simple_dither_with_method(255.0, &r2);

    let golden_result: Vec<f32> = test_values
        .par_iter()
        .copied()
        .simple_dither_with_method(255.0, &golden);

    println!("Hash method (first 5):   {:?}", &hash_result[..5]);
    println!("R2 method (first 5):     {:?}", &r2_result[..5]);
    println!("Golden method (first 5): {:?}", &golden_result[..5]);

    // Demonstrate deterministic nature
    println!("\n--- Deterministic Parallel Processing ---");
    let result1: Vec<f32> =
        values.par_iter().copied().simple_dither(255.0, 123);

    let result2: Vec<f32> =
        values.par_iter().copied().simple_dither(255.0, 123);

    assert_eq!(result1, result2);
    println!("Same seed produces identical results even in parallel!");
}
