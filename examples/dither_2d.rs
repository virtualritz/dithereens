use dithereens::*;

fn main() {
    println!("2D Dithering Methods Example\n");

    // Create a small 2D image (4x3)
    let width = 4;
    let height = 3;
    let image_data: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.4, // Row 0
        0.5, 0.6, 0.7, 0.8, // Row 1
        0.9, 0.2, 0.4, 0.6, // Row 2
    ];

    println!("Original image data ({}x{}):", width, height);
    print_image(&image_data, width);

    // Test different 2D methods
    println!("\n=== 2D Dithering Methods ===\n");

    // Interleaved Gradient Noise
    println!("--- Interleaved Gradient Noise ---");
    let mut ign_data = image_data.clone();
    let ign = InterleavedGradientNoise::new(42);
    // Using the new trait method syntax
    ign.simple_dither_slice_2d::<1, 0, _>(&mut ign_data, width, 255.0);
    print_image(&ign_data, width);

    // Spatial Hash
    println!("\n--- Spatial Hash ---");
    let mut spatial_data = image_data.clone();
    let spatial = SpatialHash::new(42);
    simple_dither_slice_2d::<1, 0, _, _>(
        &mut spatial_data,
        width,
        255.0,
        &spatial,
    );
    print_image(&spatial_data, width);

    // Blue Noise approximation
    println!("\n--- Blue Noise Approximation ---");
    let mut blue_data = image_data.clone();
    let blue_approx = BlueNoiseApprox::new(42);
    simple_dither_slice_2d::<1, 0, _, _>(
        &mut blue_data,
        width,
        255.0,
        &blue_approx,
    );
    print_image(&blue_data, width);

    // Real Blue Noise Table
    #[cfg(feature = "blue-noise")]
    {
        println!("\n--- Blue Noise Table (Real) ---");
        let mut blue_table_data = image_data.clone();
        let blue_noise = BlueNoise::new(42);
        simple_dither_slice_2d::<1, 0, _, _>(
            &mut blue_table_data,
            width,
            255.0,
            &blue_noise,
        );
        print_image(&blue_table_data, width);
    }

    // Compare single pixel across methods
    println!("\n--- Single Pixel Comparison (x=1, y=1) ---");
    let test_value = 0.5f32;
    let x = 1;
    let y = 1;
    let seed = 42;

    let ign = InterleavedGradientNoise::new(seed);
    let spatial = SpatialHash::new(seed);
    let blue_approx = BlueNoiseApprox::new(seed);

    // You can use either the free function or the trait method
    let ign_pixel = simple_dither_2d(test_value, 255.0, x, y, &ign);
    let spatial_pixel = spatial.simple_dither_2d(test_value, 255.0, x, y); // Using trait method
    let blue_approx_pixel =
        blue_approx.simple_dither_2d(test_value, 255.0, x, y); // Using trait method

    println!("Input value:    {}", test_value);
    println!("IGN:            {:.1}", ign_pixel);
    println!("Spatial:        {:.1}", spatial_pixel);
    println!("Blue Approx:    {:.1}", blue_approx_pixel);

    #[cfg(feature = "blue-noise")]
    {
        let blue_noise = BlueNoise::new(seed);
        let blue_noise_pixel =
            simple_dither_2d(test_value, 255.0, x, y, &blue_noise);
        println!("Blue Noise:     {:.1}", blue_noise_pixel);
    }

    // Show spatial distribution pattern
    println!("\n--- Spatial Distribution Pattern (8x8 grid) ---");
    println!("All pixels set to 0.5, showing dither pattern only:\n");

    let pattern_size = 8;
    let pattern = vec![0.5f32; pattern_size * pattern_size];

    println!("IGN Pattern:");
    let mut ign_pattern = pattern.clone();
    let ign = InterleavedGradientNoise::new(0);
    simple_dither_slice_2d::<1, 0, _, _>(
        &mut ign_pattern,
        pattern_size,
        255.0,
        &ign,
    );
    print_pattern(&ign_pattern, pattern_size);

    println!("\nSpatial Hash Pattern:");
    let mut spatial_pattern = pattern.clone();
    let spatial = SpatialHash::new(0);
    simple_dither_slice_2d::<1, 0, _, _>(
        &mut spatial_pattern,
        pattern_size,
        255.0,
        &spatial,
    );
    print_pattern(&spatial_pattern, pattern_size);

    println!("\nBlue Noise Approx Pattern:");
    let mut blue_pattern = pattern.clone();
    let blue_approx = BlueNoiseApprox::new(0);
    simple_dither_slice_2d::<1, 0, _, _>(
        &mut blue_pattern,
        pattern_size,
        255.0,
        &blue_approx,
    );
    print_pattern(&blue_pattern, pattern_size);

    #[cfg(feature = "blue-noise")]
    {
        println!("\nBlue Noise Table Pattern (Real):");
        let mut blue_table_pattern = pattern.clone();
        let blue_noise = BlueNoise::new(0);
        simple_dither_slice_2d::<1, 0, _, _>(
            &mut blue_table_pattern,
            pattern_size,
            255.0,
            &blue_noise,
        );
        print_pattern(&blue_table_pattern, pattern_size);
    }

    // Show seed variation
    println!("\n--- Seed Variation (IGN with different seeds) ---");
    for seed in [0, 42, 123] {
        let mut seed_test = pattern.clone();
        let ign = InterleavedGradientNoise::new(seed);
        simple_dither_slice_2d::<1, 0, _, _>(
            &mut seed_test,
            pattern_size,
            255.0,
            &ign,
        );
        println!("Seed {}:", seed);
        print_pattern(&seed_test, pattern_size);
        println!();
    }

    #[cfg(feature = "blue-noise")]
    {
        println!(
            "\n--- Seed Variation (Blue Noise Table with different seeds) ---"
        );
        for seed in [0, 42, 123] {
            let mut seed_test = pattern.clone();
            let blue_noise = BlueNoise::new(seed);
            simple_dither_slice_2d::<1, 0, _, _>(
                &mut seed_test,
                pattern_size,
                255.0,
                &blue_noise,
            );
            println!("Seed {}:", seed);
            print_pattern(&seed_test, pattern_size);
            println!();
        }
    }
}

fn print_image(data: &[f32], width: usize) {
    for (i, value) in data.iter().enumerate() {
        if i % width == 0 && i > 0 {
            println!();
        }
        print!("{:6.1} ", value);
    }
    println!();
}

fn print_pattern(data: &[f32], width: usize) {
    for (i, value) in data.iter().enumerate() {
        if i % width == 0 && i > 0 {
            println!();
        }
        // Convert to 0-9 scale for visualization
        let normalized = ((*value / 255.0) * 9.0).round() as u8;
        print!("{} ", normalized);
    }
    println!();
}
