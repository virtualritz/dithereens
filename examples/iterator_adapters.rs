use dithereens::*;

fn main() {
    println!("Iterator Adapter Examples with Deterministic Dithering\n");

    // Basic usage with iterator adapter
    let values = vec![0.5f32; 20];

    // Using the iterator adapter trait (automatically tracks indices)
    let result: Vec<f32> = values.iter().copied().simple_dither(255.0, 42);

    println!(
        "Basic iterator adapter:\n(0.5 quantized to 0..255 is sometimes 127 and sometimes 128)"
    );
    println!("  Input:  {:?}", values);
    println!("  Output: {:?}\n", result);

    // Processing with map operations before dithering - all in one chain!
    let data = vec![2.0f32, 2.0, 2.0, 4.0];
    let result: Vec<f32> = data
        .iter()
        .copied()
        // Normalize to 0-1 range
        .map(|x| x / 4.0)
        // Dither directly
        .simple_dither(255.0, 42);

    println!("Normalized then dithered:");
    println!("  Input:  {:?}", data);
    println!("  Normalized and dithered: {:?}\n", result);

    // Full dither with custom parameters using iterator adapter
    let values = vec![0.1f32, 0.3, 0.7, 0.9];
    let result: Vec<f32> = values.iter().copied().dither(50.0, 200.0, 0.8, 42);

    println!("Full dither with custom range (50-200) and strength (0.8):");
    println!("  Input:  {:?}", values);
    println!("  Output: {:?}\n", result);

    // Processing image-like scanline data
    let image_data: Vec<Vec<f32>> = vec![
        vec![0.1, 0.3, 0.5],
        vec![0.7, 0.9, 0.2],
        vec![0.4, 0.6, 0.8],
    ];

    let processed: Vec<Vec<f32>> = image_data
        .into_iter()
        .enumerate()
        .map(|(row_idx, row)| {
            let exposed: Vec<f32> = row
                .into_iter()
                // +3/4 EV exposure
                .map(|pixel| pixel * 2.0f32.powf(3.0 / 4.0))
                .collect();
            // Dither with unique seed per row for variation
            exposed
                .into_iter()
                .simple_dither(255.0, 42 + row_idx as u32)
        })
        .collect();

    println!("2D image data processing:");
    for (i, row) in processed.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }
    println!();

    // Demonstrating deterministic nature with iterator adapters
    let values = vec![0.2f32, 0.5, 0.8];
    let result1: Vec<f32> = values.iter().copied().simple_dither(255.0, 123);
    let result2: Vec<f32> = values.iter().copied().simple_dither(255.0, 123);

    println!("Deterministic results (same seed):");
    println!("  First run:  {:?}", result1);
    println!("  Second run: {:?}", result2);
    println!("  Are equal: {}\n", result1 == result2);

    // Different seeds produce different results
    let result3: Vec<f32> = values.iter().copied().simple_dither(255.0, 456);
    println!("Different seed:");
    println!("  Seed 123: {:?}", result1);
    println!("  Seed 456: {:?}", result3);
    println!("  Are different: {}", result1 != result3);

    // Show using different methods with iterator adapters
    println!("\n--- Using Different Methods with Iterator Adapters ---");
    let test = vec![0.5f32; 5];

    let hash = Hash::new(42);
    let r2 = R2::new(42);
    let golden = GoldenRatio::new(42);

    let hash_result: Vec<f32> =
        test.iter().copied().simple_dither_with_method(255.0, &hash);
    let r2_result: Vec<f32> =
        test.iter().copied().simple_dither_with_method(255.0, &r2);
    let golden_result: Vec<f32> = test
        .iter()
        .copied()
        .simple_dither_with_method(255.0, &golden);

    println!("Hash:   {:?}", hash_result);
    println!("R2:     {:?}", r2_result);
    println!("Golden: {:?}", golden_result);

    // Complex iterator chain
    println!("\n--- Complex Iterator Chain ---");
    let complex_result: Vec<f32> = (0..10)
        .map(|i| i as f32 / 10.0)
        .filter(|&x| x > 0.2)
        .map(|x| x.sqrt())
        .simple_dither(255.0, 42);

    println!("Filtered and transformed: {:?}", complex_result);
}
