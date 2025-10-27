#![cfg_attr(feature = "nightly_f16", feature(f16))]

use dithereens::*;

fn main() {
    println!("Dithereens - Float-to-Float Precision Dithering\n");
    println!("==============================================\n");

    // Example 1: f64 to f32 conversion
    println!("--- f64 to f32 Conversion ---\n");

    let value_f64: f64 = 1.234567890123456;
    println!("Original f64: {:.15}", value_f64);

    // Without dithering (simple cast)
    let undithered: f32 = value_f64 as f32;
    println!("Undithered cast: {:.15}", undithered as f64);

    // With dithering
    let dithered: f32 = dither_float(value_f64, 0, 42);
    println!("Dithered: {:.15}", dithered as f64);
    println!();

    // Example 2: Gradient conversion (HDR scenario)
    println!("--- Gradient Conversion (HDR Sky Example) ---\n");
    println!("Converting smooth gradient from f64 to f32:");

    let gradient_values: Vec<f64> =
        (0..10).map(|i| 1.0 + i as f64 * 0.000001).collect();

    println!("\nOriginal f64 values:");
    for (i, &val) in gradient_values.iter().enumerate() {
        println!("  [{}]: {:.10}", i, val);
    }

    // Undithered conversion
    let undithered_gradient: Vec<f32> =
        gradient_values.iter().map(|&v| v as f32).collect();

    println!("\nUndithered f32 (direct cast):");
    for (i, &val) in undithered_gradient.iter().enumerate() {
        println!("  [{}]: {:.10}", i, val);
    }

    // Dithered conversion
    let dithered_gradient: Vec<f32> = dither_float_slice(&gradient_values, 42);

    println!("\nDithered f32:");
    for (i, &val) in dithered_gradient.iter().enumerate() {
        println!("  [{}]: {:.10}", i, val);
    }

    // Show quantization effect.
    let unique_undithered: std::collections::HashSet<_> =
        undithered_gradient.iter().map(|&v| v.to_bits()).collect();
    let unique_dithered: std::collections::HashSet<_> =
        dithered_gradient.iter().map(|&v| v.to_bits()).collect();

    println!(
        "\nUnique values: undithered={}, dithered={}",
        unique_undithered.len(),
        unique_dithered.len()
    );
    println!();

    // Example 3: Different dithering methods
    println!("--- Comparing Dithering Methods ---\n");

    let test_value: f64 = 2.718281828459045;
    println!("Test value (f64): {:.15}", test_value);

    let hash = Hash::new(42);
    let r2 = R2::new(42);
    let golden = GoldenRatio::new(42);

    let result_hash: f32 = dither_float_with(test_value, 0, &hash);
    let result_r2: f32 = dither_float_with(test_value, 0, &r2);
    let result_golden: f32 = dither_float_with(test_value, 0, &golden);

    println!("Hash method: {:.15}", result_hash);
    println!("R2 method: {:.15}", result_r2);
    println!("GoldenRatio method: {:.15}", result_golden);
    println!();

    // Example 4: 2D dithering for images
    println!("--- 2D Image Dithering ---\n");

    let width = 8;
    let height = 8;
    let image_data: Vec<f64> = vec![1.5; width * height];

    println!("Image size: {}x{}", width, height);
    println!("Uniform value: 1.5");

    let ign = InterleavedGradientNoise::new(42);
    let spatial = SpatialHash::new(42);

    let dithered_ign: Vec<f32> =
        dither_float_slice_2d(&image_data, width, &ign);
    let dithered_spatial: Vec<f32> =
        dither_float_slice_2d(&image_data, width, &spatial);

    println!("\nWith InterleavedGradientNoise (first 8 values):");
    for (i, &val) in dithered_ign.iter().take(8).enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }

    println!("\nWith SpatialHash (first 8 values):");
    for (i, &val) in dithered_spatial.iter().take(8).enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }
    println!();

    // Example 5: Deterministic behavior
    println!("--- Deterministic Behavior ---\n");

    let seed = 777;
    let test_values: Vec<f64> = vec![1.1, 2.2, 3.3];

    let result1: Vec<f32> = dither_float_slice(&test_values, seed);
    let result2: Vec<f32> = dither_float_slice(&test_values, seed);

    println!("Same seed produces identical results:");
    println!("Run 1: {:?}", result1);
    println!("Run 2: {:?}", result2);
    println!("Equal: {}", result1 == result2);
    println!();

    // Example 6: Using trait methods
    println!("--- Using Trait Methods ---\n");

    let value: f64 = 6.28318;
    let method = R2::new(999);

    let result: f32 = method.dither_float(value, 0);

    println!("Value: {}", value);
    println!("Dithered with R2 trait method: {}", result);
    println!();

    // f16 examples (only with nightly_f16 feature)
    #[cfg(feature = "nightly_f16")]
    {
        println!("--- f32 to f16 Conversion (nightly_f16) ---\n");

        let value_f32: f32 = 3.14159;
        println!("Original f32: {}", value_f32);

        let undithered_f16: f16 = value_f32 as f16;
        println!("Undithered f16: {}", undithered_f16 as f32);

        let dithered_f16: f16 = dither_float(value_f32, 0, 42);
        println!("Dithered f16: {}", dithered_f16 as f32);
        println!();

        println!("--- f16 Gradient (Smooth Sky) ---\n");

        let sky_gradient: Vec<f32> =
            (0..20).map(|i| 10.0 + i as f32 * 0.1).collect();

        let undithered_sky: Vec<f16> =
            sky_gradient.iter().map(|&v| v as f16).collect();
        let dithered_sky: Vec<f16> = dither_float_slice(&sky_gradient, 42);

        let unique_undithered_f16: std::collections::HashSet<_> =
            undithered_sky.iter().map(|&v| v.to_bits()).collect();
        let unique_dithered_f16: std::collections::HashSet<_> =
            dithered_sky.iter().map(|&v| v.to_bits()).collect();

        println!("Sky gradient 20 values:");
        println!("Unique undithered: {}", unique_undithered_f16.len());
        println!("Unique dithered: {}", unique_dithered_f16.len());
        println!(
            "Dithering increases unique values by {}",
            unique_dithered_f16.len() - unique_undithered_f16.len()
        );
        println!();
    }

    #[cfg(not(feature = "nightly_f16"))]
    {
        println!("--- f16 Examples (requires nightly_f16 feature) ---\n");
        println!("Recompile with --features nightly_f16 to see f16 examples.");
        println!();
    }

    println!("==============================================");
    println!("Float precision dithering demonstration complete!");
}
