use dithereens::*;

#[test]
fn test_dither_basic() {
    // Test deterministic dithering with index and seed
    for index in 0..100 {
        let value = dither(0.5_f32, 0.0, 255.0, 0.5, index, 42);
        let clamped = value.clamp(0.0, 255.0) as u8;
        assert!(clamped >= 126 && clamped <= 129); // Allow small range due to dithering
    }
}

#[test]
fn test_simple_dither_basic() {
    // Test simple dither with deterministic seed
    for index in 0..100 {
        let value = simple_dither(0.5_f32, 255.0, index, 42) as u8;
        assert!(value == 127 || value == 128);
    }
}

#[test]
fn test_dither_edge_cases() {
    // Test with 0.0
    let result = dither(0.0_f32, 0.0, 255.0, 0.5, 0, 42);
    assert!(result >= -0.5 && result <= 0.5);

    // Test with 1.0
    let result = dither(1.0_f32, 0.0, 255.0, 0.5, 0, 42);
    assert!(result >= 254.5 && result <= 255.5);

    // Test with negative min
    let result = dither(0.5_f32, -100.0, 100.0, 0.5, 0, 42);
    assert!(result >= -0.5 && result <= 0.5);
}

#[test]
fn test_simple_dither_edge_cases() {
    // Test with 0.0
    let result = simple_dither(0.0_f32, 255.0, 0, 42);
    assert!(result >= 0.0 && result <= 255.0);

    // Test with 1.0
    let result = simple_dither(1.0_f32, 255.0, 0, 42);
    assert!(result >= 0.0 && result <= 255.0);

    // Test with values outside [0,1] range
    let result = simple_dither(-0.5_f32, 255.0, 0, 42);
    assert!(result >= 0.0 && result <= 255.0);

    let result = simple_dither(1.5_f32, 255.0, 0, 42);
    assert!(result >= 0.0 && result <= 255.0);
}

#[test]
fn test_dither_deterministic_with_seed() {
    // Same index and seed should produce same results
    let result1 = dither(0.5_f32, 0.0, 255.0, 0.5, 10, 42);
    let result2 = dither(0.5_f32, 0.0, 255.0, 0.5, 10, 42);
    assert_eq!(result1, result2);

    // Different seeds should produce some different results
    // With amplitude 0.5, values round to either 127 or 128, so we expect some
    // collisions
    let mut different_count = 0;
    for i in 0..20 {
        let r1 = dither(0.5_f32, 0.0, 255.0, 0.5, i, 42);
        let r2 = dither(0.5_f32, 0.0, 255.0, 0.5, i, 100 + i); // Use more different seeds
        if r1 != r2 {
            different_count += 1;
        }
    }
    // With amplitude 0.5 and only 2 possible output values (127 or 128),
    // we expect roughly 50% to be different (allowing for hash bias)
    assert!(
        different_count >= 5,
        "Different seeds should produce at least some different results, got {}/20",
        different_count
    );

    // Different indices usually produce different results
    // Test with more spread out indices to ensure we hit different hash values
    let result4 = dither(0.5_f32, 0.0, 255.0, 0.5, 15, 42);
    let result5 = dither(0.5_f32, 0.0, 255.0, 0.5, 20, 42);
    let result6 = dither(0.5_f32, 0.0, 255.0, 0.5, 25, 42);
    // At least one should be different (indices 15 and 19 produce 127, others
    // 128)
    assert!(
        result1 != result4 || result1 != result5 || result1 != result6,
        "Different indices should produce different results"
    );
}

#[test]
fn test_batch_functions() {
    let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];

    let dithered = dither_iter(values.clone(), 0.0, 255.0, 0.5, 42);
    assert_eq!(dithered.len(), 5);

    let simple_dithered = simple_dither_iter(values, 255.0, 42);
    assert_eq!(simple_dithered.len(), 5);

    // All results should be within expected ranges
    for &result in &simple_dithered {
        assert!(result >= 0.0 && result <= 255.0);
    }
}

#[test]
fn test_inplace_functions() {
    let mut values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let original_values = values.clone();

    dither_slice(&mut values, 0.0, 255.0, 0.5, 42);

    // Values should have changed
    assert_ne!(values, original_values);
    assert_eq!(values.len(), 5);

    let mut values2 = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    simple_dither_slice(&mut values2, 255.0, 42);

    // All results should be within expected ranges
    for &result in &values2 {
        assert!(result >= 0.0 && result <= 255.0);
    }
}

#[test]
fn test_different_numeric_types() {
    // Test f64
    let result_f64 = dither(0.5_f64, 0.0, 255.0, 0.5, 0, 42);
    assert!(result_f64 >= 126.5 && result_f64 <= 128.5);

    // Test simple_dither with f64
    let result_simple_f64 = simple_dither(0.5_f64, 255.0, 0, 42);
    assert!(result_simple_f64 >= 0.0 && result_simple_f64 <= 255.0);
}

#[test]
fn test_zero_dither_amplitude() {
    let result = dither(0.5_f32, 0.0, 255.0, 0.0, 0, 42);
    assert_eq!(result, 128.0); // Should be exactly the expected value without dithering
}

#[test]
fn test_different_methods() {
    let value = 0.5_f32;
    let seed = 42;

    // Test different 1D methods
    let hash_method = Hash::new(seed);
    let r2_method = R2::new(seed);
    let golden_method = GoldenRatio::new(seed);
    let hash_result =
        dither_with_linear_rng(value, 0.0, 255.0, 0.5, 0, &hash_method);
    let r2_result =
        dither_with_linear_rng(value, 0.0, 255.0, 0.5, 0, &r2_method);
    let golden_result =
        dither_with_linear_rng(value, 0.0, 255.0, 0.5, 0, &golden_method);

    // All should produce valid results
    assert!(hash_result >= 126.5 && hash_result <= 128.5);
    assert!(r2_result >= 126.5 && r2_result <= 128.5);
    assert!(golden_result >= 126.5 && golden_result <= 128.5);

    // But they should be different (different algorithms)
    // Note: They might occasionally be the same by chance, but usually
    // different
}

#[test]
fn test_2d_methods() {
    let value = 0.5_f32;
    let x = 10;
    let y = 20;
    let seed = 42;

    // Test different 2D methods
    let ign_method = InterleavedGradientNoise::new(seed);
    let spatial_method = SpatialHash::new(seed);
    let ign_result = dither_2d(value, 0.0, 255.0, 0.5, x, y, &ign_method);
    let spatial_result =
        dither_2d(value, 0.0, 255.0, 0.5, x, y, &spatial_method);

    // All should produce valid results
    assert!(ign_result >= 126.5 && ign_result <= 128.5);
    assert!(spatial_result >= 126.5 && spatial_result <= 128.5);

    #[cfg(feature = "blue_noise")]
    {
        let blue_method = BlueNoise::new(seed);
        let blue_result = dither_2d(value, 0.0, 255.0, 0.5, x, y, &blue_method);
        assert!(blue_result >= 126.5 && blue_result <= 128.5);
    }
}

#[test]
fn test_2d_slice_functions() {
    let width = 4;
    let mut values = vec![0.5f32; width * 3]; // 4x3 image
    let original = values.clone();

    let ign_method = InterleavedGradientNoise::new(42);
    dither_slice_2d(&mut values, width, 0.0, 255.0, 0.5, &ign_method);

    // Should have changed
    assert_ne!(values, original);

    // All values should be in valid range
    for &v in &values {
        assert!(v >= 126.5 && v <= 128.5);
    }
}

#[test]
fn test_simple_2d_functions() {
    let width = 4;
    let mut values = vec![0.5f32; width * 3]; // 4x3 image

    let spatial_method = SpatialHash::new(42);
    simple_dither_slice_2d(&mut values, width, 255.0, &spatial_method);

    // All values should be in valid range
    for &v in &values {
        assert!(v >= 0.0 && v <= 255.0);
        // More specifically, around 127-128 for 0.5 input
        assert!(v >= 126.0 && v <= 129.0);
    }
}

#[test]
fn test_batch_consistency() {
    let values = vec![0.2, 0.5, 0.8];

    // Test that same seed produces same results
    let results1 = simple_dither_iter(values.clone(), 255.0, 42);
    let results2 = simple_dither_iter(values.clone(), 255.0, 42);
    assert_eq!(results1, results2);

    // Different seed should produce different results
    // Use seed 200 which produces different results for value 0.5
    let results3 = simple_dither_iter(values.clone(), 255.0, 200);
    assert_ne!(
        results1, results3,
        "Different seeds should produce different results"
    );
}

#[test]
fn test_iterator_functions() {
    // Test with different input types
    let vec_input = vec![0.2f32, 0.5, 0.8];
    let array_input = [0.2f32, 0.5, 0.8];
    let slice_input = &[0.2f32, 0.5, 0.8][..];

    // All should work with IntoIterator API
    let _result1 = dither_iter(vec_input.clone(), 0.0, 255.0, 0.5, 42);
    let _result2 = dither_iter(array_input, 0.0, 255.0, 0.5, 42);
    let _result3 =
        dither_iter(slice_input.iter().copied(), 0.0, 255.0, 0.5, 42);

    // Iterator chains work too
    let _result4 =
        dither_iter((0..10).map(|i| i as f32 / 10.0), 0.0, 255.0, 0.5, 42);

    // Test simple_dither_iter
    let _result5 = simple_dither_iter(vec_input, 255.0, 42);
    let _result6 = simple_dither_iter(array_input, 255.0, 42);
}

#[test]
fn test_method_consistency() {
    // Test that using default method is same as explicitly using Hash
    let value = 0.5_f32;

    let default_result = dither(value, 0.0, 255.0, 0.5, 10, 42);
    let hash_method = Hash::new(42);
    let hash_result =
        dither_with_linear_rng(value, 0.0, 255.0, 0.5, 10, &hash_method);

    assert_eq!(default_result, hash_result);
}

#[test]
fn test_slice_with_methods() {
    let mut values1 = vec![0.2, 0.5, 0.8];
    let mut values2 = values1.clone();

    // Test with different methods produces different results
    let hash_method = Hash::new(42);
    let r2_method = R2::new(42);
    simple_dither_slice_with_linear_rng(&mut values1, 255.0, &hash_method);
    simple_dither_slice_with_linear_rng(&mut values2, 255.0, &r2_method);

    // Both should be valid
    for &v in &values1 {
        assert!(v >= 0.0 && v <= 255.0);
    }
    for &v in &values2 {
        assert!(v >= 0.0 && v <= 255.0);
    }

    // But different (usually - might occasionally match by chance)
    // We don't assert inequality as it could fail randomly
}
