#[cfg(feature = "rayon")]
mod rayon_tests {
    use dithereens::DitherParallelIteratorExt;
    use rayon::prelude::*;

    #[test]
    fn test_parallel_simple_dither() {
        let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];

        let result: Vec<f32> =
            values.par_iter().copied().simple_dither(255.0, 42);

        assert_eq!(result.len(), 5);
        // All values should be in valid range [0.0, 255.0]
        for &value in &result {
            assert!(value >= 0.0 && value <= 255.0);
        }
    }

    #[test]
    fn test_parallel_dither() {
        let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];

        let result: Vec<f32> =
            values.par_iter().copied().dither(0.0, 255.0, 0.5, 42);

        assert_eq!(result.len(), 5);
        // Values should be reasonable (not necessarily clamped like
        // simple_dither)
        for &value in &result {
            assert!(value >= -1.0 && value <= 256.0); // Allow some tolerance for dithering
        }
    }

    #[test]
    fn test_parallel_with_map_chain() {
        let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];

        let result: Vec<f32> = values
            .par_iter()
            .copied()
            .map(|x| x * 0.5) // Scale down first
            .simple_dither(255.0, 42);

        assert_eq!(result.len(), 5);
        for &value in &result {
            assert!(value >= 0.0 && value <= 255.0);
        }
    }

    #[test]
    fn test_parallel_vs_sequential_consistency() {
        let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];

        // Sequential version using iter adapter
        use dithereens::DitherIteratorExt;
        let seq_result: Vec<f32> =
            values.iter().copied().simple_dither(255.0, 42);

        // Parallel version
        let par_result: Vec<f32> =
            values.par_iter().copied().simple_dither(255.0, 42);

        // Both should produce identical results with deterministic dithering
        assert_eq!(
            seq_result, par_result,
            "Sequential and parallel versions should produce identical results"
        );
    }

    #[test]
    fn test_parallel_with_different_seeds() {
        let values = vec![0.5f32; 10];

        let result1: Vec<f32> =
            values.par_iter().copied().simple_dither(255.0, 42);

        let result2: Vec<f32> =
            values.par_iter().copied().simple_dither(255.0, 43);

        // Same seed should give same results
        let result3: Vec<f32> =
            values.par_iter().copied().simple_dither(255.0, 42);

        assert_eq!(
            result1, result3,
            "Same seed should produce identical results"
        );
        assert_ne!(
            result1, result2,
            "Different seeds should produce different results"
        );
    }

    #[test]
    fn test_parallel_with_methods() {
        use dithereens::{GoldenRatio, Hash, R2};
        let values = vec![0.5f32; 10];

        // Test with different 1D methods
        let hash_method = Hash::new(42);
        let r2_method = R2::new(42);
        let golden_method = GoldenRatio::new(42);

        let hash_result: Vec<f32> = values
            .par_iter()
            .copied()
            .simple_dither_with(255.0, &hash_method);

        let r2_result: Vec<f32> = values
            .par_iter()
            .copied()
            .simple_dither_with(255.0, &r2_method);

        let golden_result: Vec<f32> = values
            .par_iter()
            .copied()
            .simple_dither_with(255.0, &golden_method);

        // All should produce valid results
        for &v in &hash_result {
            assert!(v >= 0.0 && v <= 255.0);
        }
        for &v in &r2_result {
            assert!(v >= 0.0 && v <= 255.0);
        }
        for &v in &golden_result {
            assert!(v >= 0.0 && v <= 255.0);
        }

        // Different methods should generally produce different results
        // (though might occasionally match by chance)
    }
}
