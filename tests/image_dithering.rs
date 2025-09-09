use dithereens::*;
use std::{
    env,
    path::{Path, PathBuf},
};

/// Apply sRGB linear to non-linear EOTF (gamma correction)
fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Create a test gradient image
fn create_test_image(width: usize, height: usize) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(width * height * 4);

    for y in 0..height {
        for x in 0..width {
            // Create a smooth gradient
            let fx = x as f32 / (width - 1) as f32;
            let fy = y as f32 / (height - 1) as f32;

            // Create interesting patterns for R, G, B
            let r = fx;
            let g = fy;
            let b = (fx * fy).sqrt();

            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
            pixels.push(1.0); // Alpha
        }
    }

    pixels
}

/// Save an image as PNG
fn save_as_png(
    path: &Path,
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut img_data = Vec::with_capacity(width * height * channels);

    for pixel in data.chunks(channels) {
        for c in 0..channels {
            let value = (pixel[c].clamp(0.0, 1.0) * 255.0) as u8;
            img_data.push(value);
        }
    }

    let color_type = match channels {
        3 => image::ColorType::Rgb8,
        4 => image::ColorType::Rgba8,
        _ => panic!("Unsupported channel count: {}", channels),
    };

    image::save_buffer(
        path,
        &img_data,
        width as u32,
        height as u32,
        color_type,
    )?;
    Ok(())
}

/// Load a PNG image for comparison
fn load_png(
    path: &Path,
) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let rgba = img.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    Ok((rgba.into_raw(), width, height))
}

/// Compare two images and return true if they match
fn images_match(img1: &[u8], img2: &[u8]) -> bool {
    if img1.len() != img2.len() {
        return false;
    }

    // Allow small differences due to potential PNG encoding variations
    for (a, b) in img1.iter().zip(img2.iter()) {
        if (*a as i32 - *b as i32).abs() > 1 {
            return false;
        }
    }
    true
}

/// Get the path for expected or output image
fn get_image_path(name: &str, is_expected: bool) -> PathBuf {
    let update_mode = env::var("UPDATE_EXPECTED").is_ok();

    if is_expected && !update_mode {
        PathBuf::from("tests/expected_images").join(format!("{}.png", name))
    } else if is_expected && update_mode {
        // In update mode, save to expected_images
        PathBuf::from("tests/expected_images").join(format!("{}.png", name))
    } else {
        // Output for comparison
        PathBuf::from("target/test_output").join(format!("{}.png", name))
    }
}

/// Save and optionally compare with expected image
fn save_and_compare(
    name: &str,
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let update_mode = env::var("UPDATE_EXPECTED").is_ok();

    // Ensure directories exist
    std::fs::create_dir_all("tests/expected_images").ok();
    std::fs::create_dir_all("target/test_output").ok();

    if update_mode {
        // Save to expected_images
        let expected_path = get_image_path(name, true);
        save_as_png(&expected_path, data, width, height, channels)?;
        println!("Updated expected image: {:?}", expected_path);
    } else {
        // Save to target and compare with expected
        let output_path = get_image_path(name, false);
        save_as_png(&output_path, data, width, height, channels)?;

        let expected_path = get_image_path(name, true);
        if expected_path.exists() {
            // Load both images and compare
            let output_img = load_png(&output_path)?;
            let expected_img = load_png(&expected_path)?;

            assert_eq!(
                output_img.1, expected_img.1,
                "Image width mismatch for {}",
                name
            );
            assert_eq!(
                output_img.2, expected_img.2,
                "Image height mismatch for {}",
                name
            );

            if !images_match(&output_img.0, &expected_img.0) {
                panic!(
                    "Image mismatch for {}. Output saved to {:?}. Run with UPDATE_EXPECTED=1 to update expected images.",
                    name, output_path
                );
            }
            println!("✓ Image {} matches expected", name);
        } else {
            println!(
                "Warning: No expected image for {}. Output saved to {:?}. Run with UPDATE_EXPECTED=1 to create expected images.",
                name, output_path
            );
        }
    }

    Ok(())
}

/// Convert RGB to grayscale using luminance weights
fn rgb_to_grayscale(r: f32, g: f32, b: f32) -> f32 {
    // Use standard luminance weights (ITU-R BT.709)
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Dither to N grayscale levels (e.g., 2 for black and white, 4 for 4 grayscale
/// levels) This converts to grayscale first, then applies dithering
fn dither_to_grayscale_levels<M: DitherMethod2D>(
    data: &mut [f32],
    width: usize,
    levels: usize,
    seed: u32,
    method: &M,
) {
    let channels = 4; // We know it's RGBA
    let height = data.len() / (width * channels);

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let pixel_index = (y * width + x) * channels;

            // Convert to grayscale first
            let r = data[pixel_index];
            let g = data[pixel_index + 1];
            let b = data[pixel_index + 2];
            let gray = rgb_to_grayscale(r, g, b);

            // Apply sRGB gamma correction
            let gamma_corrected = linear_to_srgb(gray);

            // Quantize to N levels with dithering
            let quantized = if levels == 2 {
                // Black and white
                let dithered = simple_dither_2d(
                    gamma_corrected,
                    1.0,
                    x as u32,
                    y as u32,
                    method,
                );
                if dithered > 0.5 { 1.0 } else { 0.0 }
            } else {
                // Multiple grayscale levels
                let scale = (levels - 1) as f32;
                let dithered = simple_dither_2d(
                    gamma_corrected,
                    scale,
                    x as u32,
                    y as u32,
                    method,
                );
                (dithered.round() / scale).clamp(0.0, 1.0)
            };

            // Set all RGB channels to the same grayscale value
            data[pixel_index] = quantized;
            data[pixel_index + 1] = quantized;
            data[pixel_index + 2] = quantized;
            // Alpha remains unchanged
        }
    }
}

/// Dither color image to N levels per channel (preserves color)
fn dither_color_to_levels<M: DitherMethod2D>(
    data: &mut [f32],
    width: usize,
    levels: usize,
    seed: u32,
    method: &M,
) {
    let channels = 4; // We know it's RGBA
    let height = data.len() / (width * channels);

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let pixel_index = (y * width + x) * channels;

            // Process RGB channels independently (preserves color)
            for c in 0..3 {
                let idx = pixel_index + c;

                // Apply sRGB gamma correction
                let linear_value = data[idx];
                let gamma_corrected = linear_to_srgb(linear_value);

                // Quantize to N levels with dithering
                // Create a method with a different seed per channel
                let channel_method =
                    InterleavedGradientNoise::new(seed + c as u32);
                let quantized = if levels == 2 {
                    // Two levels per channel
                    let dithered = simple_dither_2d(
                        gamma_corrected,
                        1.0,
                        x as u32,
                        y as u32,
                        &channel_method,
                    );
                    if dithered > 0.5 { 1.0 } else { 0.0 }
                } else {
                    // Multiple levels per channel
                    let scale = (levels - 1) as f32;
                    let dithered = simple_dither_2d(
                        gamma_corrected,
                        scale,
                        x as u32,
                        y as u32,
                        &channel_method,
                    );
                    (dithered.round() / scale).clamp(0.0, 1.0)
                };

                data[idx] = quantized;
            }
        }
    }
}

#[test]
fn test_gradient_dithering_black_and_white() {
    let width = 256;
    let height = 256;

    // Create a test gradient image
    let mut pixels = create_test_image(width, height);

    // Apply black and white dithering (converts to grayscale)
    let method = InterleavedGradientNoise::new(42);
    dither_to_grayscale_levels(&mut pixels, width, 2, 42, &method);

    // Save and compare
    save_and_compare("gradient_bw", &pixels, width, height, 4)
        .expect("Failed to save/compare image");

    // Verify the output contains only 0.0 and 1.0 values (for RGB channels)
    for chunk in pixels.chunks(4) {
        for c in 0..3 {
            let val = chunk[c];
            assert!(
                val == 0.0 || val == 1.0,
                "Expected only 0.0 or 1.0, got {}",
                val
            );
        }
    }
}

#[test]
fn test_gradient_dithering_four_levels() {
    let width = 256;
    let height = 256;

    // Create a test gradient image
    let mut pixels = create_test_image(width, height);

    // Apply 4-level grayscale dithering
    let method = InterleavedGradientNoise::new(42);
    dither_to_grayscale_levels(&mut pixels, width, 4, 42, &method);

    // Save and compare
    save_and_compare("gradient_4levels", &pixels, width, height, 4)
        .expect("Failed to save/compare image");

    // Verify the output contains only the expected 4 levels
    let expected_levels = vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    for chunk in pixels.chunks(4) {
        for c in 0..3 {
            let val = chunk[c];
            let is_valid = expected_levels
                .iter()
                .any(|&level| (val - level).abs() < 0.01);
            assert!(
                is_valid,
                "Unexpected value: {}, expected one of {:?}",
                val, expected_levels
            );
        }
    }
}

#[test]
fn test_gradient_dithering_comparison() {
    let width = 256;
    let height = 256;

    // Create a test gradient image
    let original_pixels = create_test_image(width, height);

    // Test different dithering methods
    // Test IGN
    {
        let mut pixels = original_pixels.clone();
        let method = InterleavedGradientNoise::new(42);
        dither_to_grayscale_levels(&mut pixels, width, 2, 42, &method);
        save_and_compare("gradient_bw_ign", &pixels, width, height, 4)
            .expect("Failed to save/compare IGN image");
    }

    // Test SpatialHash
    {
        let mut pixels = original_pixels.clone();
        let method = SpatialHash::new(42);
        dither_to_grayscale_levels(&mut pixels, width, 2, 42, &method);
        save_and_compare("gradient_bw_spatialhash", &pixels, width, height, 4)
            .expect("Failed to save/compare SpatialHash image");
    }

    // Test BlueNoise
    #[cfg(feature = "blue_noise")]
    {
        let mut pixels = original_pixels.clone();
        let method = BlueNoise::new(42);
        dither_to_grayscale_levels(&mut pixels, width, 2, 42, &method);
        save_and_compare("gradient_bw_bluenoise", &pixels, width, height, 4)
            .expect("Failed to save/compare BlueNoise image");
    }
}

#[test]
fn test_gradient_dithering_different_levels() {
    let width = 256;
    let height = 256;

    // Create a test gradient image
    let original_pixels = create_test_image(width, height);

    // Test different numbers of levels
    for levels in [2, 3, 4, 8, 16] {
        let mut pixels = original_pixels.clone();

        let method = InterleavedGradientNoise::new(42);
        dither_to_grayscale_levels(&mut pixels, width, levels, 42, &method);

        // Save and compare
        let name = format!("gradient_{}levels", levels);
        save_and_compare(&name, &pixels, width, height, 4)
            .expect("Failed to save/compare image");
    }
}

#[test]
fn test_photo_simulation() {
    let width = 320;
    let height = 240;

    // Create a more complex test image simulating a photo
    let mut pixels = Vec::with_capacity(width * height * 4);

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // Create a more complex pattern with smooth gradients and features
            let cx = (fx - 0.5).abs() * 2.0;
            let cy = (fy - 0.5).abs() * 2.0;
            let dist = (cx * cx + cy * cy).sqrt().min(1.0);

            let r = (1.0 - dist) * (0.5 + 0.5 * (fx * 10.0).sin());
            let g = (1.0 - dist) * (0.5 + 0.5 * (fy * 10.0).sin());
            let b = (1.0 - dist) * (0.5 + 0.5 * ((fx + fy) * 10.0).sin());

            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
            pixels.push(1.0);
        }
    }

    // Save original
    save_and_compare("photo_sim_original", &pixels, width, height, 4)
        .expect("Failed to save/compare original");

    // Apply different dithering levels
    #[cfg(feature = "blue_noise")]
    {
        for levels in [2, 4, 8] {
            let mut dithered = pixels.clone();
            let method = BlueNoise::new(42);
            dither_to_grayscale_levels(
                &mut dithered,
                width,
                levels,
                42,
                &method,
            );

            let name = format!("photo_sim_{}levels", levels);
            save_and_compare(&name, &dithered, width, height, 4)
                .expect("Failed to save/compare dithered image");
        }
    }

    #[cfg(not(feature = "blue_noise"))]
    {
        for levels in [2, 4, 8] {
            let mut dithered = pixels.clone();
            let method = InterleavedGradientNoise::new(42);
            dither_to_grayscale_levels(
                &mut dithered,
                width,
                levels,
                42,
                &method,
            );

            let name = format!("photo_sim_{}levels", levels);
            save_and_compare(&name, &dithered, width, height, 4)
                .expect("Failed to save/compare dithered image");
        }
    }
}

#[test]
fn test_dithering_consistency() {
    // Test that dithering produces consistent results
    let width = 128;
    let height = 128;

    let pixels1 = create_test_image(width, height);
    let pixels2 = pixels1.clone();

    let mut dithered1 = pixels1.clone();
    let mut dithered2 = pixels2.clone();

    // Apply same dithering with same parameters
    let method = InterleavedGradientNoise::new(42);
    dither_to_grayscale_levels(&mut dithered1, width, 4, 42, &method);
    dither_to_grayscale_levels(&mut dithered2, width, 4, 42, &method);

    // Results should be identical
    assert_eq!(dithered1, dithered2, "Dithering should be deterministic");

    // Save for visual inspection
    save_and_compare("consistency_test", &dithered1, width, height, 4)
        .expect("Failed to save/compare consistency test");
}

#[test]
fn test_color_vs_grayscale_dithering() {
    // Test the difference between color and grayscale dithering
    let width = 256;
    let height = 256;

    let original = create_test_image(width, height);

    // Apply grayscale dithering (black and white)
    let mut grayscale_bw = original.clone();
    let method = InterleavedGradientNoise::new(42);
    dither_to_grayscale_levels(&mut grayscale_bw, width, 2, 42, &method);
    save_and_compare("grayscale_bw", &grayscale_bw, width, height, 4)
        .expect("Failed to save/compare grayscale B&W");

    // Apply color dithering (2 levels per channel - produces 8 possible colors)
    let mut color_2level = original.clone();
    let method = InterleavedGradientNoise::new(42);
    dither_color_to_levels(&mut color_2level, width, 2, 42, &method);
    save_and_compare("color_2level", &color_2level, width, height, 4)
        .expect("Failed to save/compare color 2-level");

    // Verify grayscale is actually grayscale
    for chunk in grayscale_bw.chunks(4) {
        assert_eq!(chunk[0], chunk[1], "R should equal G for grayscale");
        assert_eq!(chunk[1], chunk[2], "G should equal B for grayscale");
        assert!(
            chunk[0] == 0.0 || chunk[0] == 1.0,
            "Should be black or white"
        );
    }

    // Verify color dithering can have different RGB values
    let mut has_color = false;
    for chunk in color_2level.chunks(4) {
        if chunk[0] != chunk[1] || chunk[1] != chunk[2] {
            has_color = true;
            break;
        }
    }
    assert!(has_color, "Color dithering should preserve some color");
}
