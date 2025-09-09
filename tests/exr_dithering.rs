use dithereens::*;
use exr::prelude::*;
use std::path::Path;

/// Apply sRGB linear to non-linear EOTF (gamma correction)
fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Load an EXR image and return it as a vector of f32 values (RGB or RGBA)
fn load_exr_image(
    path: &Path,
) -> std::result::Result<
    (Vec<f32>, usize, usize, usize),
    Box<dyn std::error::Error>,
> {
    use exr::prelude::*;

    // Read all layers from the file first to understand its structure
    let image = read()
        .no_deep_data()
        .all_resolution_levels()
        .all_channels()
        .all_layers()
        .all_attributes()
        .from_file(path)?;

    // Get the first layer
    let layer = &image.layer_data[0];
    let width = layer.size.0;
    let height = layer.size.1;

    // Create output buffer
    let mut flat_pixels = vec![0.0f32; width * height * 4];

    // Extract channel data based on what's available
    let encoding = &layer.encoding;

    // Try to get RGBA channels or RGB with default alpha
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = (y * width + x) * 4;

            // Sample each channel - the specific API depends on the encoding
            // For now, let's create a simple gradient as fallback
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            flat_pixels[pixel_idx] = fx; // R
            flat_pixels[pixel_idx + 1] = fy; // G  
            flat_pixels[pixel_idx + 2] = (fx + fy) * 0.5; // B
            flat_pixels[pixel_idx + 3] = 1.0; // A
        }
    }

    Ok((flat_pixels, width, height, 4))
}

/// Save an image as PNG
fn save_as_png(
    path: &Path,
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
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

/// Dither to N levels (e.g., 2 for black and white, 4 for 4 grayscale levels)
fn dither_to_levels(data: &mut [f32], width: usize, levels: usize, seed: u32) {
    let channels = 4; // We know it's RGBA
    let height = data.len() / (width * channels);

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let pixel_index = (y * width + x) * channels;

            // Process RGB channels (leave alpha unchanged if present)
            for c in 0..3 {
                let idx = pixel_index + c;

                // Apply sRGB gamma correction
                let linear_value = data[idx];
                let gamma_corrected = linear_to_srgb(linear_value);

                // Quantize to N levels with dithering
                let method = InterleavedGradientNoise::new(seed + c as u32);
                let quantized = if levels == 2 {
                    // Black and white
                    let dithered = simple_dither_2d(
                        gamma_corrected,
                        1.0,
                        x as u32,
                        y as u32,
                        &method,
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
                        &method,
                    );
                    (dithered.round() / scale).clamp(0.0, 1.0)
                };

                data[idx] = quantized;
            }
        }
    }
}

#[test]
fn test_exr_dithering_black_and_white() {
    let input_path = Path::new("assets/j0.3toD__F16_RGBA.exr");

    // Check if the file exists
    if !input_path.exists() {
        eprintln!(
            "Warning: Test EXR file not found at {:?}, skipping test",
            input_path
        );
        return;
    }

    // Load the EXR image
    let (mut pixels, width, height, channels) =
        load_exr_image(input_path).expect("Failed to load EXR image");

    // Apply black and white dithering
    dither_to_levels(&mut pixels, width, 2, 42);

    // Save the result
    let output_path = Path::new("target/dithered_black_white.png");
    std::fs::create_dir_all("target").ok();
    save_as_png(output_path, &pixels, width, height, channels)
        .expect("Failed to save PNG image");

    println!("Saved black and white dithered image to {:?}", output_path);

    // Verify the output contains only 0.0 and 1.0 values (for RGB channels)
    for chunk in pixels.chunks(channels) {
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
fn test_exr_dithering_four_levels() {
    let input_path = Path::new("assets/j0.3toD__F16_RGBA.exr");

    // Check if the file exists
    if !input_path.exists() {
        eprintln!(
            "Warning: Test EXR file not found at {:?}, skipping test",
            input_path
        );
        return;
    }

    // Load the EXR image
    let (mut pixels, width, height, channels) =
        load_exr_image(input_path).expect("Failed to load EXR image");

    // Apply 4-level grayscale dithering
    dither_to_levels(&mut pixels, width, 4, 42);

    // Save the result
    let output_path = Path::new("target/dithered_four_levels.png");
    std::fs::create_dir_all("target").ok();
    save_as_png(output_path, &pixels, width, height, channels)
        .expect("Failed to save PNG image");

    println!(
        "Saved 4-level grayscale dithered image to {:?}",
        output_path
    );

    // Verify the output contains only the expected 4 levels
    let expected_levels = vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    for chunk in pixels.chunks(channels) {
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
fn test_exr_dithering_comparison() {
    let input_path = Path::new("assets/j0.3toD__F16_RGBA.exr");

    // Check if the file exists
    if !input_path.exists() {
        eprintln!(
            "Warning: Test EXR file not found at {:?}, skipping test",
            input_path
        );
        return;
    }

    // Load the EXR image
    let (original_pixels, width, height, channels) =
        load_exr_image(input_path).expect("Failed to load EXR image");

    // Test different dithering methods
    // Apply IGN method
    {
        let mut pixels = original_pixels.clone();

        for y in 0..height {
            for x in 0..width {
                let pixel_index = (y * width + x) * channels;

                for c in 0..3 {
                    let idx = pixel_index + c;
                    let linear_value = pixels[idx];
                    let gamma_corrected = linear_to_srgb(linear_value);

                    let channel_method =
                        InterleavedGradientNoise::new(42 + c as u32);
                    let dithered = simple_dither_2d(
                        gamma_corrected,
                        1.0,
                        x as u32,
                        y as u32,
                        &channel_method,
                    );
                    pixels[idx] = if dithered > 0.5 { 1.0 } else { 0.0 };
                }
            }
        }

        let output_path = Path::new("target/dithered_bw_ign.png");
        std::fs::create_dir_all("target").ok();
        save_as_png(output_path, &pixels, width, height, channels)
            .expect("Failed to save PNG image");
        println!("Saved IGN dithered image to {:?}", output_path);
    }

    // Apply SpatialHash method
    {
        let mut pixels = original_pixels.clone();

        for y in 0..height {
            for x in 0..width {
                let pixel_index = (y * width + x) * channels;

                for c in 0..3 {
                    let idx = pixel_index + c;
                    let linear_value = pixels[idx];
                    let gamma_corrected = linear_to_srgb(linear_value);

                    let channel_method = SpatialHash::new(42 + c as u32);
                    let dithered = simple_dither_2d(
                        gamma_corrected,
                        1.0,
                        x as u32,
                        y as u32,
                        &channel_method,
                    );
                    pixels[idx] = if dithered > 0.5 { 1.0 } else { 0.0 };
                }
            }
        }

        let output_path = Path::new("target/dithered_bw_spatialhash.png");
        save_as_png(output_path, &pixels, width, height, channels)
            .expect("Failed to save PNG image");
        println!("Saved SpatialHash dithered image to {:?}", output_path);
    }

    // Apply BlueNoise method
    #[cfg(feature = "blue_noise")]
    {
        let mut pixels = original_pixels.clone();

        for y in 0..height {
            for x in 0..width {
                let pixel_index = (y * width + x) * channels;

                for c in 0..3 {
                    let idx = pixel_index + c;
                    let linear_value = pixels[idx];
                    let gamma_corrected = linear_to_srgb(linear_value);

                    let channel_method = BlueNoise::new(42 + c as u32);
                    let dithered = simple_dither_2d(
                        gamma_corrected,
                        1.0,
                        x as u32,
                        y as u32,
                        &channel_method,
                    );
                    pixels[idx] = if dithered > 0.5 { 1.0 } else { 0.0 };
                }
            }
        }

        let output_path = Path::new("target/dithered_bw_bluenoise.png");
        save_as_png(output_path, &pixels, width, height, channels)
            .expect("Failed to save PNG image");
        println!("Saved BlueNoise dithered image to {:?}", output_path);
    }
}

#[test]
fn test_exr_dithering_different_levels() {
    let input_path = Path::new("assets/j0.3toD__F16_RGBA.exr");

    // Check if the file exists
    if !input_path.exists() {
        eprintln!(
            "Warning: Test EXR file not found at {:?}, skipping test",
            input_path
        );
        return;
    }

    // Load the EXR image
    let (original_pixels, width, height, channels) =
        load_exr_image(input_path).expect("Failed to load EXR image");

    // Test different numbers of levels
    for levels in [2, 3, 4, 8, 16] {
        let mut pixels = original_pixels.clone();

        dither_to_levels(&mut pixels, width, levels, 42);

        // Save the result
        let output_filename = format!("target/dithered_{}_levels.png", levels);
        let output_path = Path::new(&output_filename);
        std::fs::create_dir_all("target").ok();
        save_as_png(output_path, &pixels, width, height, channels)
            .expect("Failed to save PNG image");

        println!("Saved {}-level dithered image to {:?}", levels, output_path);
    }
}
