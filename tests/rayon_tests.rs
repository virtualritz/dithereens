#[cfg(feature = "rayon")]
mod rayon_tests {
    use dithereens::*;
    use rand::{
        SeedableRng,
        rngs::{SmallRng, StdRng},
    };

    // Test RNG that satisfies Rayon trait bounds: Rng + Sync + Send + Clone
    #[derive(Clone)]
    struct TestRng {
        inner: StdRng,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self {
                inner: StdRng::seed_from_u64(seed),
            }
        }
    }

    impl rand::RngCore for TestRng {
        fn next_u32(&mut self) -> u32 {
            self.inner.next_u32()
        }

        fn next_u64(&mut self) -> u64 {
            self.inner.next_u64()
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            self.inner.fill_bytes(dest)
        }
    }

    // Rng is automatically implemented for types that implement RngCore

    // TestRng automatically implements Send + Sync + Clone

    #[test]
    fn test_rayon_dither_iter() {
        let values = vec![0.2f32, 0.5, 0.8];
        let rng = TestRng::new(42);

        // Test that Rayon batch function works
        let results = dither_iter(values, 0.0, 255.0, 0.5, &rng);
        assert_eq!(results.len(), 3);

        // All results should be within reasonable range
        for &result in &results {
            assert!(result >= -1.0 && result <= 256.0);
        }
    }

    #[test]
    fn test_rayon_simple_dither_iter() {
        let values = vec![0.2f32, 0.5, 0.8];
        let rng = TestRng::new(42);

        // Test that Rayon simple batch function works
        let results = simple_dither_iter(values, 255.0, &rng);
        assert_eq!(results.len(), 3);

        // All results should be within expected ranges
        for &result in &results {
            assert!(result >= 0.0 && result <= 255.0);
        }
    }

    #[test]
    fn test_rayon_dither_slice() {
        let mut values = vec![0.2f32, 0.5, 0.8];
        let original_values = values.clone();
        let rng = TestRng::new(42);

        dither_slice(&mut values, 0.0, 255.0, 0.5, &rng);

        // Values should have changed (with high probability)
        assert_eq!(values.len(), 3);
        // At least one value should be different due to dithering
        assert!(
            values != original_values || values.iter().all(|&x| x == x.round())
        );
    }

    #[test]
    fn test_rayon_simple_dither_slice() {
        let mut values = vec![0.2f32, 0.5, 0.8];
        let original_values = values.clone();
        let rng = TestRng::new(42);

        simple_dither_slice(&mut values, 255.0, &rng);

        // Values should have changed and be within range
        assert_eq!(values.len(), 3);
        for &result in &values {
            assert!(result >= 0.0 && result <= 255.0);
        }
        // At least one value should be different due to dithering
        assert!(
            values != original_values || values.iter().all(|&x| x == x.round())
        );
    }

    #[test]
    fn test_rayon_deterministic_results() {
        let values = vec![0.1f32, 0.3, 0.7, 0.9];

        // Test deterministic behavior with same seed
        let rng1 = TestRng::new(123);
        let results1 = simple_dither_iter(values.clone(), 255.0, &rng1);

        let rng2 = TestRng::new(123);
        let results2 = simple_dither_iter(values.clone(), 255.0, &rng2);

        assert_eq!(
            results1, results2,
            "Results should be deterministic with same seed"
        );
    }

    #[test]
    fn test_rayon_large_dataset() {
        // Test with larger dataset to verify parallel processing works
        let values: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let rng = TestRng::new(42);

        let results = simple_dither_iter(values, 255.0, &rng);
        assert_eq!(results.len(), 1000);

        // All results should be within expected ranges
        for &result in &results {
            assert!(result >= 0.0 && result <= 255.0);
        }
    }

    #[test]
    fn test_rayon_vs_sequential_consistency() {
        // Compare results between rayon and sequential versions
        // Note: Due to parallel processing, results may differ,
        // but both should produce valid dithered values
        let values = vec![0.25f32, 0.5, 0.75];

        // Sequential version (without rayon)
        let mut seq_rng = SmallRng::seed_from_u64(42);
        let seq_results = values
            .iter()
            .map(|&v| simple_dither(v, 255.0, &mut seq_rng))
            .collect::<Vec<_>>();

        // Parallel version (with rayon)
        let par_rng = TestRng::new(42);
        let par_results = simple_dither_iter(values, 255.0, &par_rng);

        // Both should have same length and valid ranges
        assert_eq!(seq_results.len(), par_results.len());

        for (&seq, &par) in seq_results.iter().zip(par_results.iter()) {
            assert!(seq >= 0.0 && seq <= 255.0);
            assert!(par >= 0.0 && par <= 255.0);
        }
    }
}
