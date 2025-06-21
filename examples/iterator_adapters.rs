use dithereens::*;
use rand::{SeedableRng, rngs::SmallRng};

fn main() {
    let mut rng = SmallRng::seed_from_u64(42);

    println!("Iterator Adapter Examples\n");

    // Basic usage with simple_dither..
    let values = vec![0.5f32; 20];
    let result: Vec<f32> =
        values.iter().copied().simple_dither(255.0, &mut rng);

    println!(
        "Basic simple_dither:\n(0.5 quantized to 0..255 is sometimes 127 and sometimes 128)"
    );
    println!("  Input:  {:?}", values);
    println!("  Output: {:?}\n", result);

    // Chaining with map operations.
    let data = vec![2.0f32, 2.0, 2.0, 4.0];
    let result: Vec<f32> = data
        .iter()
        .copied()
        // Normalize to 0-1 range.
        .map(|x| x / 4.0)
        // Dither.
        .simple_dither(255.0, &mut rng);

    println!("Chained with map:");
    println!("  Input:  {:?}", data);
    println!("  Normalized and dithered: {:?}\n", result);

    // Full dither with custom parameters.
    let values = vec![0.1f32, 0.3, 0.7, 0.9];
    let result: Vec<f32> = values
        .iter()
        .copied()
        // Custom range and dither strength.
        .dither(50.0, 200.0, 0.8, &mut rng);

    println!("Full dither with custom range (50-200) and strength (0.8):");
    println!("  Input:  {:?}", values);
    println!("  Output: {:?}\n", result);

    // Processing image-like scanline data.
    let image_data: Vec<Vec<f32>> = vec![
        vec![0.1, 0.3, 0.5],
        vec![0.7, 0.9, 0.2],
        vec![0.4, 0.6, 0.8],
    ];

    let processed: Vec<Vec<f32>> = image_data
        .into_iter()
        .map(|row| {
            row.into_iter()
                // +3/4 EV exposure.
                .map(|pixel| pixel * 2.0f32.powf(3.0 / 4.0))
                // Dither.
                .simple_dither(255.0, &mut rng)
        })
        .collect();

    println!("2D image data processing:");
    for (i, row) in processed.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }
    println!();

    // Old way (still available).
    let values = vec![0.2f32, 0.5, 0.8];
    let old_way =
        simple_dither_iter(values.iter().map(|&x| x * 0.8), 255.0, &mut rng);

    // New way with iterator adapter.
    let new_way: Vec<f32> = values
        .iter()
        .copied()
        .map(|x| x * 0.8)
        .simple_dither(255.0, &mut rng);

    assert_eq!(old_way, new_way);
}
