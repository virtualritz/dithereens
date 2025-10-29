#[cfg(feature = "rayon")]
mod rayon_tests {
    use dithereens::*;

    #[test]
    fn test_rayon_dither_iter() {
        let values = vec![0.2f32, 0.5, 0.8];

        // Test that Rayon batch function works with deterministic seed
        let results = dither_iter(values, 0.0, 255.0, 0.5, 42);
        assert_eq!(results.len(), 3);

        // All results should be within reasonable range
        for &result in &results {
            assert!((-1.0..=256.0).contains(&result));
        }
    }

    #[test]
    fn test_rayon_simple_dither_iter() {
        let values = vec![0.2f32, 0.5, 0.8];

        // Test that Rayon simple batch function works with deterministic seed
        let results = simple_dither_iter(values, 255.0, 42);
        assert_eq!(results.len(), 3);

        // All results should be within expected ranges
        for &result in &results {
            assert!((0.0..=255.0).contains(&result));
        }
    }

    #[test]
    fn test_rayon_dither_slice() {
        let mut values = vec![0.2f32, 0.5, 0.8];
        let original_values = values.clone();

        dither_slice(&mut values, 0.0, 255.0, 0.5, 42);

        // Values should have changed
        assert_eq!(values.len(), 3);
        // Values should be different from original due to dithering
        assert_ne!(values, original_values);

        // All values should be properly dithered
        for &v in &values {
            assert!((0.0..=255.0).contains(&v));
        }
    }

    #[test]
    fn test_rayon_simple_dither_slice() {
        let mut values = vec![0.2f32, 0.5, 0.8];
        let original_values = values.clone();

        simple_dither_slice(&mut values, 255.0, 42);

        // Values should have changed and be within range
        assert_eq!(values.len(), 3);
        for &result in &values {
            assert!((0.0..=255.0).contains(&result));
        }
        // Values should be different from original due to dithering
        assert_ne!(values, original_values);
    }

    #[test]
    fn test_rayon_deterministic_results() {
        let values = vec![0.1f32, 0.3, 0.7, 0.9];

        // Test deterministic behavior with same seed
        let results1 = simple_dither_iter(values.clone(), 255.0, 123);
        let results2 = simple_dither_iter(values.clone(), 255.0, 123);

        assert_eq!(
            results1, results2,
            "Results should be deterministic with same seed"
        );

        // Different seed should produce different results
        let results3 = simple_dither_iter(values, 255.0, 456);
        assert_ne!(
            results1, results3,
            "Different seeds should produce different results"
        );
    }

    #[test]
    fn test_rayon_large_dataset() {
        // Test with larger dataset to verify parallel processing works
        let values: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();

        let results = simple_dither_iter(values, 255.0, 42);
        assert_eq!(results.len(), 1000);

        // All results should be within expected ranges
        for &result in &results {
            assert!((0.0..=255.0).contains(&result));
        }
    }

    #[test]
    fn test_rayon_vs_sequential_consistency() {
        // Compare results between rayon and sequential versions
        // With deterministic index-based dithering, results should be identical
        let values = vec![0.25f32, 0.5, 0.75];

        // Sequential version (using iterator adapter for fair comparison)
        let seq_results: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| simple_dither(v, 255.0, i as u32, 42))
            .collect();

        // Parallel version (with rayon)
        let par_results = simple_dither_iter(values, 255.0, 42);

        // Both should produce identical results with deterministic dithering
        assert_eq!(
            seq_results, par_results,
            "Sequential and parallel versions should produce identical results"
        );

        // All values should be valid
        for &v in &par_results {
            assert!((0.0..=255.0).contains(&v));
        }
    }
}
