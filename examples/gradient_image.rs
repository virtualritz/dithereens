//! This produces the gradient image for the README.

use dithereens::simple_dither_slice;
use image::{ImageBuffer, Rgba};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate RGB f32 image with darkish gradient.
    // This is the image we'll process with dithering.
    //
    // The data is non-linear sRGB but still in 0.0..=1.0 range after this.
    let mut image_data = generate_gradient_image();

    let no_dither_result = process_without_dither(&image_data);

    let dither_result = process_with_dither(&mut image_data);

    // Create combined image with transparent separator
    create_combined_image(&no_dither_result, &dither_result)?;

    Ok(())
}

/// Generate a 1024x64 RGB f32 image with a dark green gradient.
fn generate_gradient_image() -> Vec<f32> {
    (0..(WIDTH * HEIGHT))
        .flat_map(|pixel_idx| {
            // Convert linear pixel index to x,y coordinates.
            let x = pixel_idx % WIDTH;

            // Create horizontal gradient from black to dark blue.
            let gradient_value = x as f32 / (WIDTH - 1) as f32;
            let v = linear_to_srgb(gradient_value * 0.03);

            [0.0, v, 0.0]
        })
        .collect()
}

/// Process image without dithering: multiply by 255 and cast to u8.
fn process_without_dither(image_data: &[f32]) -> Vec<u8> {
    image_data
        .iter()
        .map(|&value| (value * 255.0) as u8)
        .collect()
}

/// Process image with dithering using `dithereens`.
fn process_with_dither(image_data: &mut [f32]) -> Vec<u8> {
    let mut rng = wyrand::WyRand::new(42);

    // Apply dithering while quantizing to 0..=255 range.
    simple_dither_slice(image_data, 255.0, &mut rng);

    // Cast to u8.
    image_data.iter().map(|&value| value as u8).collect()
}

/// Convert linear RGB to sRGB (gamma correction).
///
/// Applies the standard sRGB transfer function to convert from linear light
/// values to the non-linear sRGB color space used by most displays and image
/// formats.
fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Create combined RGBA image with both gradients and 1px transparent
/// separator line.
fn create_combined_image(
    no_dither_data: &[u8],
    dither_data: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    // Total height: 2 gradients + 1 pixel separator
    let combined_height = HEIGHT * 2 + 1;
    let mut rgba_data =
        Vec::with_capacity((WIDTH * combined_height * 4) as usize);

    // Add no-dither gradient (top)
    for i in (0..no_dither_data.len()).step_by(3) {
        rgba_data.push(no_dither_data[i]); // R
        rgba_data.push(no_dither_data[i + 1]); // G
        rgba_data.push(no_dither_data[i + 2]); // B
        rgba_data.push(255); // A (opaque)
    }

    // Add transparent separator line
    for _ in 0..WIDTH {
        rgba_data.extend_from_slice(&[0, 0, 0, 0]); // Fully transparent pixel
    }

    // Add dithered gradient (bottom)
    for i in (0..dither_data.len()).step_by(3) {
        rgba_data.push(dither_data[i]); // R
        rgba_data.push(dither_data[i + 1]); // G
        rgba_data.push(dither_data[i + 2]); // B
        rgba_data.push(255); // A (opaque)
    }

    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(WIDTH, combined_height, rgba_data)
            .ok_or("Failed to create image buffer")?;

    img.save("before_after_dither.png")?;

    Ok(())
}
