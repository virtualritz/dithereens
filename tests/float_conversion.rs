// Tests for float-to-float precision conversions with dithering.

#![cfg_attr(feature = "nightly_f16", feature(f16))]

use dithereens::*;

#[test]
fn f64_to_f32_conversion() {
    let value: f64 = 1.234567890123456;
    let result: f32 = dither_float(value, 0, 42);

    // Result should be close to the original value.
    assert!((result as f64 - value).abs() < 0.001);
}

#[test]
fn f64_to_f32_deterministic() {
    let value: f64 = 1.234567890123456;
    let seed = 777;

    // Same seed and index should produce same result.
    let result1: f32 = dither_float(value, 0, seed);
    let result2: f32 = dither_float(value, 0, seed);

    assert_eq!(result1, result2);
}

#[test]
fn f64_to_f32_with_different_methods() {
    let value: f64 = 1.234567890123456;

    let hash = Hash::new(42);
    let r2 = R2::new(42);
    let golden = GoldenRatio::new(42);

    let result_hash: f32 = dither_float_with(value, 0, &hash);
    let result_r2: f32 = dither_float_with(value, 0, &r2);
    let result_golden: f32 = dither_float_with(value, 0, &golden);

    // All should be close to original.
    assert!((result_hash as f64 - value).abs() < 0.001);
    assert!((result_r2 as f64 - value).abs() < 0.001);
    assert!((result_golden as f64 - value).abs() < 0.001);
}

#[test]
fn f64_to_f32_slice() {
    let values: Vec<f64> = vec![1.1, 1.2, 1.3, 1.4, 1.5];
    let result: Vec<f32> = dither_float_slice(&values, 42);

    assert_eq!(result.len(), values.len());

    for (i, (&orig, &dithered)) in values.iter().zip(result.iter()).enumerate()
    {
        assert!(
            (dithered as f64 - orig).abs() < 0.001,
            "Index {}: expected ~{}, got {}",
            i,
            orig,
            dithered
        );
    }
}

#[test]
fn f64_to_f32_slice_deterministic() {
    let values: Vec<f64> = vec![1.1, 1.2, 1.3, 1.4, 1.5];
    let seed = 123;

    let result1: Vec<f32> = dither_float_slice(&values, seed);
    let result2: Vec<f32> = dither_float_slice(&values, seed);

    assert_eq!(result1, result2);
}

#[cfg(feature = "rayon")]
#[test]
fn f64_to_f32_parallel_vs_sequential() {
    let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let seed = 999;

    let result: Vec<f32> = dither_float_slice(&values, seed);

    // Results should be deterministic.
    assert_eq!(result.len(), values.len());

    // Same results on re-run.
    let result2: Vec<f32> = dither_float_slice(&values, seed);
    assert_eq!(result, result2);
}

#[test]
fn f64_to_f32_2d_with_ign() {
    let value: f64 = 1.5;
    let method = InterleavedGradientNoise::new(42);

    let result: f32 = dither_float_2d(value, 10, 20, &method);

    assert!((result as f64 - value).abs() < 0.01);
}

#[test]
fn f64_to_f32_2d_with_spatial_hash() {
    let value: f64 = 2.75;
    let method = SpatialHash::new(42);

    let result: f32 = dither_float_2d(value, 15, 25, &method);

    assert!((result as f64 - value).abs() < 0.01);
}

#[test]
fn f64_to_f32_slice_2d() {
    let width = 16;
    let height = 16;
    let values: Vec<f64> = vec![1.5; width * height];
    let method = InterleavedGradientNoise::new(42);

    let result: Vec<f32> = dither_float_slice_2d(&values, width, &method);

    assert_eq!(result.len(), values.len());

    for &dithered in &result {
        assert!((dithered as f64 - 1.5).abs() < 0.01);
    }
}

#[test]
fn f64_to_f32_special_values() {
    let inf = f64::INFINITY;
    let neg_inf = f64::NEG_INFINITY;
    let nan = f64::NAN;

    let result_inf: f32 = dither_float(inf, 0, 42);
    let result_neg_inf: f32 = dither_float(neg_inf, 0, 42);
    let result_nan: f32 = dither_float(nan, 0, 42);

    assert!(result_inf.is_infinite() && result_inf.is_sign_positive());
    assert!(result_neg_inf.is_infinite() && result_neg_inf.is_sign_negative());
    assert!(result_nan.is_nan());
}

#[test]
fn f64_to_f32_out_of_range() {
    // Value larger than f32::MAX.
    let large_value: f64 = 1e100;
    let result: f32 = dither_float(large_value, 0, 42);

    // Should saturate to infinity.
    assert!(result.is_infinite());
}

#[cfg(feature = "nightly_f16")]
mod f16_tests {
    use super::*;

    #[test]
    fn f32_to_f16_conversion() {
        let value: f32 = 1.234;
        let result: f16 = dither_float(value, 0, 42);

        // Result should be close to the original value.
        assert!((result as f32 - value).abs() < 0.01);
    }

    #[test]
    fn f32_to_f16_deterministic() {
        let value: f32 = 1.234;
        let seed = 777;

        let result1: f16 = dither_float(value, 0, seed);
        let result2: f16 = dither_float(value, 0, seed);

        assert_eq!(result1, result2);
    }

    #[test]
    fn f64_to_f16_conversion() {
        let value: f64 = 1.234567;
        let result: f16 = dither_float(value, 0, 42);

        assert!((result as f64 - value).abs() < 0.01);
    }

    #[test]
    fn f32_to_f16_with_different_methods() {
        let value: f32 = 2.5;

        let hash = Hash::new(42);
        let r2 = R2::new(42);
        let golden = GoldenRatio::new(42);

        let result_hash: f16 = dither_float_with(value, 0, &hash);
        let result_r2: f16 = dither_float_with(value, 0, &r2);
        let result_golden: f16 = dither_float_with(value, 0, &golden);

        // All should be close to original.
        assert!((result_hash as f32 - value).abs() < 0.1);
        assert!((result_r2 as f32 - value).abs() < 0.1);
        assert!((result_golden as f32 - value).abs() < 0.1);
    }

    #[test]
    fn f32_to_f16_slice() {
        let values: Vec<f32> = vec![1.1, 1.2, 1.3, 1.4, 1.5];
        let result: Vec<f16> = dither_float_slice(&values, 42);

        assert_eq!(result.len(), values.len());

        for (&orig, &dithered) in values.iter().zip(result.iter()) {
            assert!((dithered as f32 - orig).abs() < 0.1);
        }
    }

    #[test]
    fn f32_to_f16_slice_deterministic() {
        let values: Vec<f32> = vec![1.1, 1.2, 1.3, 1.4, 1.5];
        let seed = 123;

        let result1: Vec<f16> = dither_float_slice(&values, seed);
        let result2: Vec<f16> = dither_float_slice(&values, seed);

        assert_eq!(result1, result2);
    }

    #[test]
    fn f32_to_f16_2d_with_ign() {
        let value: f32 = 10.5;
        let method = InterleavedGradientNoise::new(42);

        let result: f16 = dither_float_2d(value, 10, 20, &method);

        assert!((result as f32 - value).abs() < 0.1);
    }

    #[test]
    fn f32_to_f16_slice_2d() {
        let width = 16;
        let height = 16;
        let values: Vec<f32> = vec![5.5; width * height];
        let method = InterleavedGradientNoise::new(42);

        let result: Vec<f16> = dither_float_slice_2d(&values, width, &method);

        assert_eq!(result.len(), values.len());

        for &dithered in &result {
            assert!((dithered as f32 - 5.5).abs() < 0.2);
        }
    }

    #[test]
    fn f32_to_f16_special_values() {
        let inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let nan = f32::NAN;

        let result_inf: f16 = dither_float(inf, 0, 42);
        let result_neg_inf: f16 = dither_float(neg_inf, 0, 42);
        let result_nan: f16 = dither_float(nan, 0, 42);

        assert!(result_inf.is_infinite() && result_inf.is_sign_positive());
        assert!(
            result_neg_inf.is_infinite() && result_neg_inf.is_sign_negative()
        );
        assert!(result_nan.is_nan());
    }

    #[test]
    fn f32_to_f16_out_of_range() {
        // f16 max is ~65504.
        let large_value: f32 = 70000.0;
        let result: f16 = dither_float(large_value, 0, 42);

        // Should saturate to infinity.
        assert!(result.is_infinite());
    }

    #[test]
    fn f32_to_f16_gradient() {
        // Test smooth gradient conversion.
        let values: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let result: Vec<f16> = dither_float_slice(&values, 42);

        // Verify monotonicity (dithering shouldn't reverse order).
        for i in 1..result.len() {
            assert!(
                result[i] >= result[i - 1]
                    || (result[i] as f32 - result[i - 1] as f32).abs() < 0.2
            );
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn f32_to_f16_parallel_vs_sequential() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let seed = 999;

        let result: Vec<f16> = dither_float_slice(&values, seed);

        assert_eq!(result.len(), values.len());

        // Same results on re-run.
        let result2: Vec<f16> = dither_float_slice(&values, seed);
        assert_eq!(result, result2);
    }

    #[cfg(feature = "blue-noise")]
    #[test]
    fn f32_to_f16_with_blue_noise() {
        let value: f32 = 3.14159;
        let method = BlueNoise::new(42);

        let result: f16 = dither_float_2d(value, 10, 20, &method);

        assert!((result as f32 - value).abs() < 0.1);
    }
}

#[test]
fn trait_methods_f64_to_f32() {
    let value: f64 = 3.14159;
    let method = Hash::new(42);

    let result: f32 = method.dither_float(value, 0);

    assert!((result as f64 - value).abs() < 0.001);
}

#[test]
fn trait_methods_slice_f64_to_f32() {
    let values: Vec<f64> = vec![1.1, 2.2, 3.3];
    let method = R2::new(42);

    let result: Vec<f32> = method.dither_float_slice(&values);

    assert_eq!(result.len(), values.len());

    for (&orig, &dithered) in values.iter().zip(result.iter()) {
        assert!((dithered as f64 - orig).abs() < 0.01);
    }
}

#[test]
fn trait_methods_2d_f64_to_f32() {
    let value: f64 = 7.5;
    let method = InterleavedGradientNoise::new(42);

    let result: f32 = method.dither_float_2d(value, 5, 10);

    assert!((result as f64 - value).abs() < 0.01);
}

#[test]
fn trait_methods_slice_2d_f64_to_f32() {
    let width = 8;
    let values: Vec<f64> = vec![4.25; width * width];
    let method = SpatialHash::new(42);

    let result: Vec<f32> = method.dither_float_slice_2d(&values, width);

    assert_eq!(result.len(), values.len());

    for &dithered in &result {
        assert!((dithered as f64 - 4.25).abs() < 0.1);
    }
}
