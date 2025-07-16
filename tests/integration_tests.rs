use dithereens::*;

#[test]
fn test_dither_basic() {
    let mut rng = wyrand::WyRand::new(42);

    for _ in 0..100 {
        let value =
            Ord::clamp(0.5_f32.dither(0.0, 255.0, 0.5, &mut rng) as u8, 0, 255);
        assert!(value == 127 || 128 == value);
    }
}

#[test]
fn test_simple_dither_basic() {
    let mut rng = wyrand::WyRand::new(42);

    for _ in 0..100 {
        let value = 0.5_f32.simple_dither(255.0, &mut rng) as u8;
        assert!(value == 127 || value == 128);
    }
}

#[test]
fn test_dither_edge_cases() {
    let mut rng = wyrand::WyRand::new(42);

    // Test with 0.0
    let result = dither(0.0_f32, 0.0, 255.0, 0.5, &mut rng);
    assert!(result >= -0.5 && result <= 0.5);

    // Test with 1.0
    let result = dither(1.0_f32, 0.0, 255.0, 0.5, &mut rng);
    assert!(result >= 254.5 && result <= 255.5);

    // Test with negative min
    let result = dither(0.5_f32, -100.0, 100.0, 0.5, &mut rng);
    assert!(result >= -0.5 && result <= 0.5);
}

#[test]
fn test_simple_dither_edge_cases() {
    let mut rng = wyrand::WyRand::new(42);

    // Test with 0.0
    let result = simple_dither(0.0_f32, 255.0, &mut rng);
    assert!(result >= 0.0 && result <= 255.0);

    // Test with 1.0
    let result = simple_dither(1.0_f32, 255.0, &mut rng);
    assert!(result >= 0.0 && result <= 255.0);

    // Test with values outside [0,1] range
    let result = simple_dither(-0.5_f32, 255.0, &mut rng);
    assert!(result >= 0.0 && result <= 255.0);

    let result = simple_dither(1.5_f32, 255.0, &mut rng);
    assert!(result >= 0.0 && result <= 255.0);
}

#[test]
fn test_dither_deterministic_with_seed() {
    let mut rng1 = wyrand::WyRand::new(42);
    let mut rng2 = wyrand::WyRand::new(42);

    let result1 = dither(0.5_f32, 0.0, 255.0, 0.5, &mut rng1);
    let result2 = dither(0.5_f32, 0.0, 255.0, 0.5, &mut rng2);

    assert_eq!(result1, result2);
}

#[test]
fn test_batch_functions() {
    let mut rng = wyrand::WyRand::new(42);
    let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];

    let dithered = dither_iter(values.clone(), 0.0, 255.0, 0.5, &mut rng);
    assert_eq!(dithered.len(), 5);

    let mut rng = wyrand::WyRand::new(42);
    let simple_dithered = simple_dither_iter(values, 255.0, &mut rng);
    assert_eq!(simple_dithered.len(), 5);

    // All results should be within expected ranges
    for &result in &simple_dithered {
        assert!(result >= 0.0 && result <= 255.0);
    }
}

#[test]
fn test_inplace_functions() {
    let mut rng = wyrand::WyRand::new(42);
    let mut values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let original_values = values.clone();

    dither_slice(&mut values, 0.0, 255.0, 0.5, &mut rng);

    // Values should have changed
    assert_ne!(values, original_values);
    assert_eq!(values.len(), 5);

    let mut rng = wyrand::WyRand::new(42);
    let mut values2 = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    simple_dither_slice(&mut values2, 255.0, &mut rng);

    // All results should be within expected ranges
    for &result in &values2 {
        assert!(result >= 0.0 && result <= 255.0);
    }
}

#[test]
fn test_different_numeric_types() {
    let mut rng = wyrand::WyRand::new(42);

    // Test f64
    let result_f64 = dither(0.5_f64, 0.0, 255.0, 0.5, &mut rng);
    assert!(result_f64 >= 126.5 && result_f64 <= 128.5);

    // Test simple_dither with f64
    let result_simple_f64 = simple_dither(0.5_f64, 255.0, &mut rng);
    assert!(result_simple_f64 >= 0.0 && result_simple_f64 <= 255.0);
}

#[test]
fn test_zero_dither_amplitude() {
    let mut rng = wyrand::WyRand::new(42);

    let result = dither(0.5_f32, 0.0, 255.0, 0.0, &mut rng);
    assert_eq!(result, 128.0); // Should be exactly the expected value without dithering
}

#[test]
fn test_trait_methods() {
    let mut rng = wyrand::WyRand::new(42);

    let value = 0.5_f32;
    let result1 = value.dither(0.0, 255.0, 0.5, &mut rng);
    let result2 = dither(value, 0.0, 255.0, 0.5, &mut rng);

    // Results should be in the same range (can't compare exact values due
    // to RNG state)
    assert!(result1 >= 126.5 && result1 <= 128.5);
    assert!(result2 >= 126.5 && result2 <= 128.5);

    let result3 = value.simple_dither(255.0, &mut rng);
    assert!(result3 >= 0.0 && result3 <= 255.0);
}

#[test]
fn test_batch_vs_individual_consistency() {
    let values = vec![0.2, 0.5, 0.8];

    // When rayon feature is enabled, parallel processing may produce
    // different results due to different RNG ordering, so we only
    // test basic properties
    #[cfg(feature = "rayon")]
    {
        let rng = wyrand::WyRand::new(42);
        let batch_results = simple_dither_iter(values.clone(), 255.0, &rng);

        // Test that results are valid (within expected range)
        assert_eq!(batch_results.len(), 3);
        for &result in &batch_results {
            assert!(result >= 0.0 && result <= 255.0);
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        // Test that batch and individual calls produce same results with
        // same seed
        let mut rng1 = wyrand::WyRand::new(42);
        let batch_results =
            simple_dither_iter(values.clone(), 255.0, &mut rng1);

        let mut rng2 = wyrand::WyRand::new(42);
        let individual_results: Vec<f32> = values
            .iter()
            .map(|&v| simple_dither(v, 255.0, &mut rng2))
            .collect();

        assert_eq!(batch_results, individual_results);
    }
}

#[test]
fn test_iterator_functions() {
    let mut rng = wyrand::WyRand::new(42);

    // Test with different input types
    let vec_input = vec![0.2f32, 0.5, 0.8];
    let array_input = [0.2f32, 0.5, 0.8];
    let slice_input = &[0.2f32, 0.5, 0.8][..];

    // All should work with IntoIterator API
    let _result1 = dither_iter(vec_input.clone(), 0.0, 255.0, 0.5, &mut rng);
    let _result2 = dither_iter(array_input, 0.0, 255.0, 0.5, &mut rng);
    let _result3 =
        dither_iter(slice_input.iter().copied(), 0.0, 255.0, 0.5, &mut rng);

    // Iterator chains work too
    let _result4 = dither_iter(
        (0..10).map(|i| i as f32 / 10.0),
        0.0,
        255.0,
        0.5,
        &mut rng,
    );

    // Test simple_dither_iter
    let _result5 = simple_dither_iter(vec_input, 255.0, &mut rng);
    let _result6 = simple_dither_iter(array_input, 255.0, &mut rng);
}

#[test]
fn test_batch_consistency() {
    let values = vec![0.2, 0.5, 0.8];

    // Test that different input types produce same results
    let mut rng1 = wyrand::WyRand::new(42);
    let vec_results = simple_dither_iter(values.clone(), 255.0, &mut rng1);

    let mut rng2 = wyrand::WyRand::new(42);
    let iter_results =
        simple_dither_iter(values.iter().copied(), 255.0, &mut rng2);

    assert_eq!(vec_results, iter_results);
}

// Tests for iterator adapters

#[test]
fn test_iterator_adapter_basic() {
    let mut rng = wyrand::WyRand::new(42);
    let values = vec![0.2f32, 0.5, 0.8];

    // Test basic iterator adapter
    let result: Vec<f32> =
        values.iter().copied().simple_dither(255.0, &mut rng);

    assert_eq!(result.len(), 3);
    for &value in &result {
        assert!(value >= 0.0 && value <= 255.0);
    }
}

#[test]
fn test_iterator_adapter_chaining() {
    let mut rng = wyrand::WyRand::new(42);
    let values = vec![0.1f32, 0.3, 0.6, 0.9];

    // Test chaining with map
    let result: Vec<f32> = values
        .iter()
        .copied()
        .map(|x| x * 0.8)
        .simple_dither(255.0, &mut rng);

    assert_eq!(result.len(), 4);
    for &value in &result {
        assert!(value >= 0.0 && value <= 255.0);
    }
}

#[test]
fn test_iterator_adapter_full_dither() {
    let mut rng = wyrand::WyRand::new(42);
    let values = vec![0.2f32, 0.5, 0.8];

    // Test full dither method
    let result: Vec<f32> =
        values.iter().copied().dither(0.0, 255.0, 0.5, &mut rng);

    assert_eq!(result.len(), 3);
    for &value in &result {
        assert!(value >= -1.0 && value <= 256.0); // Allow for dither range
    }
}

#[test]
fn test_iterator_adapter_vs_function() {
    let values = vec![0.2f32, 0.5, 0.8];

    // Without rayon, results should be identical with same seed
    #[cfg(not(feature = "rayon"))]
    {
        let mut rng1 = wyrand::WyRand::new(42);
        let adapter_result: Vec<f32> =
            values.iter().copied().simple_dither(255.0, &mut rng1);

        let mut rng2 = wyrand::WyRand::new(42);
        let function_result =
            simple_dither_iter(values.iter().copied(), 255.0, &mut rng2);

        assert_eq!(adapter_result, function_result);
    }

    // With rayon, just test validity
    #[cfg(feature = "rayon")]
    {
        let rng = wyrand::WyRand::new(42);
        let adapter_result: Vec<f32> =
            values.iter().copied().simple_dither(255.0, &rng);

        assert_eq!(adapter_result.len(), 3);
        for &value in &adapter_result {
            assert!(value >= 0.0 && value <= 255.0);
        }
    }
}

#[test]
fn test_iterator_adapter_complex_chain() {
    let mut rng = wyrand::WyRand::new(42);

    // Test complex iterator chain
    let result: Vec<f32> = (0..10)
        .map(|i| i as f32 / 10.0)
        .filter(|&x| x > 0.2)
        .map(|x| x.sqrt())
        .simple_dither(255.0, &mut rng);

    assert!(result.len() > 0);
    for &value in &result {
        assert!(value >= 0.0 && value <= 255.0);
    }
}
