use dithereens::*;

fn main() {
    println!("Dithereens - Deterministic Dithering Methods\n");

    // Test values
    let values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];
    println!("Original values: {:?}", values);

    println!("\n=== 1D Dithering Methods ===\n");

    // Hash method (default)
    println!("--- Hash Method (Default) ---");
    let hash_result = simple_dither_iter(values.clone(), 255.0, 42);
    println!("Result: {:?}", hash_result);

    // R2 low-discrepancy sequence
    println!("\n--- R2 Low-Discrepancy Sequence ---");
    let r2 = R2::new(42);
    let r2_result = simple_dither_iter_with_method(values.clone(), 255.0, &r2);
    println!("Result: {:?}", r2_result);

    // Golden Ratio sequence
    println!("\n--- Golden Ratio Sequence ---");
    let golden = GoldenRatio::new(42);
    let golden_result =
        simple_dither_iter_with_method(values.clone(), 255.0, &golden);
    println!("Result: {:?}", golden_result);

    // Compare methods
    println!("\n--- Method Comparison ---");
    println!("Hash:         {:?}", hash_result);
    println!("R2:           {:?}", r2_result);
    println!("Golden Ratio: {:?}", golden_result);

    // Demonstrate deterministic nature
    println!("\n--- Reproducibility Test ---");
    println!("Running with same seed always produces the same results:");
    let test_value = 0.5f32;
    for run in 0..3 {
        let result = simple_dither(test_value, 255.0, 0, 777);
        println!("  Run {}: {} -> {}", run + 1, test_value, result);
    }

    // In-place slice dithering with different methods
    println!("\n--- In-place Slice Dithering ---");
    let mut slice_hash = values.clone();
    let mut slice_r2 = values.clone();
    let mut slice_golden = values.clone();

    simple_dither_slice(&mut slice_hash, 255.0, 123);
    let r2_123 = R2::new(123);
    let golden_123 = GoldenRatio::new(123);
    simple_dither_slice_with_method(&mut slice_r2, 255.0, &r2_123);
    simple_dither_slice_with_method(&mut slice_golden, 255.0, &golden_123);

    println!("Original: {:?}", values);
    println!("Hash:     {:?}", slice_hash);
    println!("R2:       {:?}", slice_r2);
    println!("Golden:   {:?}", slice_golden);

    // Custom dither parameters
    println!("\n--- Custom Dither Parameters ---");
    let r2_999 = R2::new(999);
    for (i, &value) in values.iter().enumerate() {
        let dithered_hash = dither(value, 0.0, 255.0, 0.5, i as u32, 999);
        let dithered_r2 =
            dither_with_method(value, 0.0, 255.0, 0.5, i as u32, &r2_999);
        println!(
            "Value {}: {:.2} -> Hash: {:.2}, R2: {:.2}",
            i, value, dithered_hash, dithered_r2
        );
    }
}
