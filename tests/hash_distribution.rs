use dithereens::*;

#[test]
fn test_hash_distribution_diagnostic() {
    println!("\n=== Testing Hash Distribution ===");

    // Test the specific case that's failing in integration_tests.rs
    let result1 = dither(0.5_f32, 0.0, 255.0, 0.5, 10, 42);
    let result4 = dither(0.5_f32, 0.0, 255.0, 0.5, 11, 42);
    let result5 = dither(0.5_f32, 0.0, 255.0, 0.5, 12, 42);
    let result6 = dither(0.5_f32, 0.0, 255.0, 0.5, 13, 42);

    println!("Index 10: {}", result1);
    println!("Index 11: {}", result4);
    println!("Index 12: {}", result5);
    println!("Index 13: {}", result6);

    println!("\nAre any different from index 10?");
    println!("11 != 10: {}", result1 != result4);
    println!("12 != 10: {}", result1 != result5);
    println!("13 != 10: {}", result1 != result6);

    let any_different =
        result1 != result4 || result1 != result5 || result1 != result6;
    println!("At least one different: {}", any_different);

    // Test with more indices
    println!("\n=== Testing more indices ===");
    for i in 10..20 {
        let result = dither(0.5_f32, 0.0, 255.0, 0.5, i, 42);
        println!("Index {}: {}", i, result);
    }

    // Test how many unique values we get
    let mut values = std::collections::HashSet::new();
    for i in 0..100 {
        let result = dither(0.5_f32, 0.0, 255.0, 0.5, i, 42);
        values.insert(result.to_bits());
    }
    println!("\nUnique values from 100 indices: {}", values.len());

    // Test seed variation
    println!("\n=== Testing seed variation ===");
    let mut different_count = 0;
    for i in 0..20 {
        let r1 = dither(0.5_f32, 0.0, 255.0, 0.5, i, 42);
        let r2 = dither(0.5_f32, 0.0, 255.0, 0.5, i, 100 + i);
        if r1 != r2 {
            different_count += 1;
        }
        println!(
            "Index {}: seed 42 = {:.2}, seed {} = {:.2}, different = {}",
            i,
            r1,
            100 + i,
            r2,
            r1 != r2
        );
    }
    println!(
        "\nDifferent results with different seeds: {}/20",
        different_count
    );
}

#[test]
fn test_batch_consistency_diagnostic() {
    println!("\n=== Testing Batch Consistency ===");

    let values = vec![0.2, 0.5, 0.8];

    // Test that same seed produces same results
    let results1 = simple_dither_iter(values.clone(), 255.0, 42);
    let results2 = simple_dither_iter(values.clone(), 255.0, 42);

    println!("Results with seed 42 (run 1): {:?}", results1);
    println!("Results with seed 42 (run 2): {:?}", results2);
    println!("Are they equal? {}", results1 == results2);

    // Different seed should produce different results
    let results3 = simple_dither_iter(values.clone(), 255.0, 43);
    println!("Results with seed 43: {:?}", results3);
    println!("Are seeds 42 and 43 different? {}", results1 != results3);

    // Check what's happening with each value
    for (i, &v) in values.iter().enumerate() {
        let r42 = simple_dither(v, 255.0, i as u32, 42);
        let r43 = simple_dither(v, 255.0, i as u32, 43);
        println!(
            "Value {} at index {}: seed 42 = {}, seed 43 = {}",
            v, i, r42, r43
        );
    }
}

#[test]
fn test_hash_method_directly() {
    println!("\n=== Testing Hash Method Directly ===");

    // Test the Hash compute method directly
    let hash = Hash::new(42);

    // Test with consecutive indices
    for i in 10..15 {
        let value = hash.compute(i);
        println!("Hash compute({}) = {}", i, value);
    }

    // Test how seed affects the hash
    println!("\nTesting seed variation:");
    for seed in [42, 43, 100, 142] {
        let hash_with_seed = Hash::new(seed);
        let value = hash_with_seed.compute(10);
        println!("Hash with seed {} compute(10) = {}", seed, value);
    }

    // Check distribution
    let mut min = 1.0f32;
    let mut max = -1.0f32;
    for i in 0..1000 {
        let value = hash.compute(i);
        min = min.min(value);
        max = max.max(value);
    }
    println!("\nHash value range from 1000 samples: [{}, {}]", min, max);
    println!("Expected range: [-1, 1]");
}
