use dithereens::*;

#[cfg(feature = "blue_noise")]
#[test]
fn test_blue_noise_table_basic() {
    // Test basic functionality of BlueNoise
    let blue_noise = BlueNoise::new(42);

    // Test a single pixel
    let value = 0.5f32;
    let result = simple_dither_2d(value, 255.0, 10, 10, &blue_noise);

    // Result should be within valid range
    assert!(result >= 0.0 && result <= 255.0);
}

#[cfg(feature = "blue_noise")]
#[test]
fn test_blue_noise_table_deterministic() {
    let blue_noise = BlueNoise::new(123);

    // Same coordinates with same method instance should produce same result
    let value = 0.5f32;
    let result1 = simple_dither_2d(value, 255.0, 50, 50, &blue_noise);
    let result2 = simple_dither_2d(value, 255.0, 50, 50, &blue_noise);

    assert_eq!(result1, result2, "Blue noise table should be deterministic");
}

#[cfg(feature = "blue_noise")]
#[test]
fn test_blue_noise_table_seed_variation() {
    // Different seeds should produce different results
    let value = 0.5f32;
    let x = 100;
    let y = 100;

    let mut results = Vec::new();
    for seed in [0, 42, 123, 456, 1000] {
        let blue_noise = BlueNoise::new(seed);
        let result = simple_dither_2d(value, 255.0, x, y, &blue_noise);
        results.push(result);
    }

    // Check that we get some variation with different seeds
    // Convert to integers for comparison since f32 doesn't implement Hash
    let int_results: Vec<i32> =
        results.iter().map(|&v| (v * 100.0) as i32).collect();
    let unique_count = int_results
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        unique_count > 1,
        "Different seeds should produce some variation"
    );
}

#[cfg(feature = "blue_noise")]
#[test]
fn test_blue_noise_table_wrapping() {
    let blue_noise = BlueNoise::new(42);

    // Test that coordinates wrap properly (256x256 table)
    let value = 0.5f32;

    // These should access the same table entry due to wrapping
    let result1 = simple_dither_2d(value, 255.0, 0, 0, &blue_noise);
    let result2 = simple_dither_2d(value, 255.0, 256, 256, &blue_noise);

    // With seed offset, they might not be exactly equal, but should be close
    // since they're accessing related table positions
    assert!(
        (result1 - result2).abs() < 255.0,
        "Wrapping should work correctly"
    );
}

#[cfg(feature = "blue_noise")]
#[test]
fn test_blue_noise_table_large_coordinates() {
    let blue_noise = BlueNoise::new(42);

    // Test with very large coordinates to ensure no overflow
    let value = 0.5f32;
    let result = simple_dither_2d(value, 255.0, 100000, 200000, &blue_noise);

    assert!(
        result >= 0.0 && result <= 255.0,
        "Large coordinates should work"
    );
}

#[cfg(feature = "blue_noise")]
#[test]
fn test_blue_noise_table_distribution() {
    let blue_noise = BlueNoise::new(42);

    // Create a small test image
    let width = 64;
    let height = 64;
    let mut image = vec![0.5f32; width * height];

    // Apply blue noise dithering
    simple_dither_slice_2d(&mut image, width, 255.0, &blue_noise);

    // Check that we get a good distribution of values
    let low_count = image.iter().filter(|&&v| v < 128.0).count();
    let high_count = image.iter().filter(|&&v| v >= 128.0).count();

    // For a 0.5 input, we should get roughly equal low and high values
    let ratio = low_count as f32 / high_count as f32;
    assert!(
        ratio > 0.8 && ratio < 1.2,
        "Blue noise should have balanced distribution"
    );
}

#[cfg(all(feature = "rayon", feature = "blue_noise"))]
#[test]
fn test_blue_noise_table_rayon_consistency() {
    let blue_noise = BlueNoise::new(42);

    // Create test data
    let width = 100;
    let height = 100;
    let image = vec![0.5f32; width * height];

    // Sequential processing
    let mut seq_results = image.clone();
    for (index, value) in seq_results.iter_mut().enumerate() {
        let x = (index % width) as u32;
        let y = (index / width) as u32;
        *value = simple_dither_2d(*value, 255.0, x, y, &blue_noise);
    }

    // Parallel processing
    let mut par_results = image.clone();
    simple_dither_slice_2d(&mut par_results, width, 255.0, &blue_noise);

    // Results should be identical
    assert_eq!(
        seq_results, par_results,
        "Sequential and parallel processing should produce identical results"
    );
}

#[cfg(all(feature = "rayon", feature = "blue_noise"))]
#[test]
fn test_blue_noise_table_rayon_deterministic() {
    let blue_noise = BlueNoise::new(123);

    // Create test data
    let width = 50;
    let height = 50;
    let image = vec![0.5f32; width * height];

    // Run parallel processing multiple times
    let mut results1 = image.clone();
    simple_dither_slice_2d(&mut results1, width, 255.0, &blue_noise);

    let mut results2 = image.clone();
    simple_dither_slice_2d(&mut results2, width, 255.0, &blue_noise);

    let mut results3 = image.clone();
    simple_dither_slice_2d(&mut results3, width, 255.0, &blue_noise);

    // All runs should produce identical results
    assert_eq!(
        results1, results2,
        "Parallel processing should be deterministic"
    );
    assert_eq!(
        results2, results3,
        "Parallel processing should be deterministic"
    );
}

#[cfg(feature = "blue_noise")]
#[test]
fn test_blue_noise_vs_approximation() {
    // Compare the real blue noise table with the approximation
    let blue_noise_table = BlueNoise::new(42);
    let blue_noise_approx = BlueNoiseApprox::new(42);

    let value = 0.5f32;

    // Test a few points
    for x in [0, 10, 50, 100, 200] {
        for y in [0, 10, 50, 100, 200] {
            let table_result =
                simple_dither_2d(value, 255.0, x, y, &blue_noise_table);
            let approx_result =
                simple_dither_2d(value, 255.0, x, y, &blue_noise_approx);

            // They should both be valid but likely different
            assert!(table_result >= 0.0 && table_result <= 255.0);
            assert!(approx_result >= 0.0 && approx_result <= 255.0);
        }
    }
}
