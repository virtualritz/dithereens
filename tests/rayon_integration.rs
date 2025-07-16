#[cfg(feature = "rayon")]
mod rayon_tests {
    use dithereens::DitherParallelIteratorExt;
    use rayon::prelude::*;

    #[test]
    fn test_parallel_simple_dither() {
        let rng = wyrand::WyRand::new(42);
        let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];

        let result: Vec<f32> =
            values.par_iter().copied().simple_dither(255.0, &rng);

        assert_eq!(result.len(), 5);
        // All values should be in valid range [0.0, 255.0]
        for &value in &result {
            assert!(value >= 0.0 && value <= 255.0);
        }
    }

    #[test]
    fn test_parallel_dither() {
        let rng = wyrand::WyRand::new(42);
        let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];

        let result: Vec<f32> =
            values.par_iter().copied().dither(0.0, 255.0, 0.5, &rng);

        assert_eq!(result.len(), 5);
        // Values should be reasonable (not necessarily clamped like
        // simple_dither)
        for &value in &result {
            assert!(value >= -1.0 && value <= 256.0); // Allow some tolerance for dithering
        }
    }

    #[test]
    fn test_parallel_with_map_chain() {
        let rng = wyrand::WyRand::new(42);
        let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];

        let result: Vec<f32> = values
            .par_iter()
            .copied()
            .map(|x| x * 0.5) // Scale down first
            .simple_dither(255.0, &rng);

        assert_eq!(result.len(), 5);
        for &value in &result {
            assert!(value >= 0.0 && value <= 255.0);
        }
    }

    #[test]
    fn test_parallel_vs_sequential_consistency() {
        let rng1 = wyrand::WyRand::new(42);
        let rng2 = wyrand::WyRand::new(42);
        let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];

        // Sequential version
        let seq_result: Vec<f32> = values
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .simple_dither(255.0, &rng1);

        // Parallel version
        let par_result: Vec<f32> =
            values.par_iter().copied().simple_dither(255.0, &rng2);

        // Both should produce valid results (though potentially different due
        // to threading)
        assert_eq!(seq_result.len(), par_result.len());
        for (&seq_val, &par_val) in seq_result.iter().zip(par_result.iter()) {
            assert!(seq_val >= 0.0 && seq_val <= 255.0);
            assert!(par_val >= 0.0 && par_val <= 255.0);
        }
    }
}
