# `dithereens`

![Before/after dithering](before_after_dither.png)
_Before (top) and after (bottom) dithering a gradient (uses `simple_dither()`, i.e. defaults)._

<!-- cargo-rdme start -->

Functions and traits for quantizing values with deterministic hash-based
error-diffusion.

Quantizing from `f64`/`f32`/`f16` to `u32`/`u16`/`u8` without dithering
creates banding. This crate provides deterministic hash-based dithering to
reduce quantization artifacts.

### Overview

- **Deterministic**: Same input with same seed always produces same output.
- **Multiple dithering methods**: Hash, R2, GoldenRatio for 1D; IGN,
  SpatialHash, BlueNoise for 2D.
- **Single values**: [`dither()`](https://docs.rs/dithereens/latest/dithereens/fn.dither.html), [`simple_dither()`](https://docs.rs/dithereens/latest/dithereens/fn.simple_dither.html).
- **Iterator processing**: [`dither_iter()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_iter.html), [`simple_dither_iter()`](https://docs.rs/dithereens/latest/dithereens/fn.simple_dither_iter.html).
- **In-place operations**: [`dither_slice()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_slice.html), [`simple_dither_slice()`](https://docs.rs/dithereens/latest/dithereens/fn.simple_dither_slice.html).
- **Image support**: Both 1D methods (processing as flat array) and 2D
  methods (using coordinates).
- **Custom methods**: Use specific dithering algorithms via
  `*_with_method()` functions.
- **`no_std` support**: Works in embedded environments.
- **Generic types**: `f32`, `f64`, `f16` (with `nightly_f16` feature), or
  any type implementing [`DitherFloat`](https://docs.rs/dithereens/latest/dithereens/trait.DitherFloat.html).
- **Blue noise**: High-quality blue noise dithering (with `blue-noise`
  feature).

### Quick Start

```rust
use dithereens::simple_dither;

let value: f32 = 0.5;

// Dither `value` to `127u8` or `128u8` deterministically.
// The same index and seed will always produce the same result.
let dithered_value: u8 =
    simple_dither(value, 255.0, 0, 42).clamp(0.0, 255.0) as u8;

assert!(dithered_value == 127 || 128 == dithered_value);
```

### Dithering Methods

#### 1D Methods (for sequential data and images as flat arrays)
- **Hash** (default): Fast hash-based dithering, good general-purpose
  quality.
- **R2**: Low-discrepancy sequence using the R2 sequence.
- **GoldenRatio**: Golden ratio-based sequence.

1D methods have been used successfully for image dithering for years by
processing images as flat arrays. They work well when you don't need
spatial correlation between pixels.

#### 2D Methods (for images using spatial coordinates)
- **InterleavedGradientNoise (IGN)**: Fast, good quality for real-time
  graphics.
- **SpatialHash**: Spatial hash function for blue noise-like properties.
- **BlueNoiseApprox**: Approximation combining IGN and SpatialHash.
- **BlueNoise** (requires `blue-noise` feature): True blue noise from
  precomputed tables.

2D methods use pixel coordinates to create spatially-aware dithering
patterns, which can produce more visually pleasing results for images.

### Using Custom Methods

```rust
use dithereens::{GoldenRatio, Hash, R2, simple_dither_with};

let value = 0.5f32;
let seed = 42;

// Use different dithering methods.
let hash_method = Hash::new(seed);
let r2_method = R2::new(seed);
let golden_method = GoldenRatio::new(seed);

let dithered_hash = simple_dither_with(value, 255.0, 0, &hash_method);
let dithered_r2 = simple_dither_with(value, 255.0, 0, &r2_method);
let dithered_golden = simple_dither_with(value, 255.0, 0, &golden_method);
```

### Dynamic Method Selection

The `LinearDither` and `SpatialDither` enums provide zero-cost dynamic
dispatch for runtime method selection:

```rust
use dithereens::{Hash, LinearDither, LinearRng, R2};

// Store different methods in a collection.
let methods: Vec<LinearDither> = vec![
    LinearDither::Hash(Hash::new(1)),
    LinearDither::R2(R2::new(2)),
];

// All methods implement LinearRng through enum_dispatch.
for method in &methods {
    let noise = method.compute(100);
    // Use the noise value...
}
```

This is useful when:
- Selecting dithering methods at runtime based on configuration.
- Storing heterogeneous collections of methods.
- Implementing plugins or extensible systems.

### Image Dithering with 1D Methods

1D methods work great for images when processed as flat arrays:

```rust
use dithereens::{Hash, simple_dither_slice};

// Example: dither a grayscale image.
let width = 256;
let height = 256;
let mut pixels: Vec<f32> = vec![0.5; width * height];

// Process entire image as flat array with 1D dithering.
simple_dither_slice(&mut pixels, 255.0, 42);

// pixels now contains dithered values.
```

### 2D Dithering for Images

2D methods use spatial coordinates for better visual results:

```rust
use dithereens::{InterleavedGradientNoise, simple_dither_slice_2d};

// Example: dither a grayscale image.
let width = 256;
let height = 256;
let mut pixels: Vec<f32> = vec![0.5; width * height];

// Use IGN for 2D dithering (1 channel, correlated noise).
let method = InterleavedGradientNoise::new(42);
simple_dither_slice_2d::<1, 0, _, _>(&mut pixels, width, 255.0, &method);

// pixels now contains dithered values.
```

### Multi-Channel Image Dithering

The 2D dithering functions support multi-channel images (RGB, RGBA) with
const-generic parameters for efficient processing:

```rust
use dithereens::{InterleavedGradientNoise, simple_dither_slice_2d};

let width = 512;
let height = 512;
let method = InterleavedGradientNoise::new(42);

// RGB image with correlated noise (same noise pattern across RGB).
// This is 3× faster than processing each channel separately.
let mut rgb_data: Vec<f32> = vec![0.5; width * height * 3];
simple_dither_slice_2d::<3, 0, _, _>(&mut rgb_data, width, 255.0, &method);

// RGB image with uncorrelated noise (different pattern per channel).
// Provides more randomness but requires computing noise per channel.
let mut rgb_data2: Vec<f32> = vec![0.5; width * height * 3];
simple_dither_slice_2d::<3, 1, _, _>(&mut rgb_data2, width, 255.0, &method);

// RGBA image with correlated noise.
let mut rgba_data: Vec<f32> = vec![0.5; width * height * 4];
simple_dither_slice_2d::<4, 0, _, _>(&mut rgba_data, width, 255.0, &method);
```

**Type parameters:**
- `CHANNELS`: Number of channels per pixel (1 = grayscale, 3 = RGB, 4 =
  RGBA).
- `SEED_OFFSET`: Per-channel noise correlation.
  - `0` = Correlated (same noise for all channels, fastest).
  - `>0` = Uncorrelated (different noise per channel, more random).

### Performance Guide

Benchmarks with 10,000 values:

- **Single values**: [`dither()`](https://docs.rs/dithereens/latest/dithereens/fn.dither.html), [`simple_dither()`](https://docs.rs/dithereens/latest/dithereens/fn.simple_dither.html).
- **In-place slice operations**: [`dither_slice()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_slice.html),
  [`simple_dither_slice()`](https://docs.rs/dithereens/latest/dithereens/fn.simple_dither_slice.html) (>5× faster than iterator methods).
- **Iterator chains**: [`dither_iter()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_iter.html), [`simple_dither_iter()`](https://docs.rs/dithereens/latest/dithereens/fn.simple_dither_iter.html), or
  [`DitherIteratorExt`](https://docs.rs/dithereens/latest/dithereens/trait.DitherIteratorExt.html) adapters (allocation overhead).

### Parallel Processing

Via `rayon` (enabled by default). With `rayon` enabled, `_iter` and
`_slice` functions use parallel processing automatically for better
performance on large datasets.

### `no_std` Support

This crate supports `no_std` environments. The `libm` crate provides a
native `round()` implementation. Without `libm`, a manual implementation is
used.

```toml
[dependencies]
# `no_std`
dithereens = { version = "0.3", default-features = false }
```

```toml
[dependencies]
# Optional: uses `libm`'s `round()` function instead of a manual
# implementation for `no_std`.
dithereens = {
   version = "0.3",
   default-features = false,
   features = ["libm"]
}
```

### Native `f16` Support

Enable the `nightly_f16` feature to use native `f16` types (requires nightly
Rust):

```toml
[dependencies]
dithereens = { version = "0.3", features = ["nightly_f16"] }
```

### Blue Noise Support

Enable the `blue-noise` feature for high-quality blue noise dithering:

```toml
[dependencies]
dithereens = { version = "0.3", features = ["blue-noise"] }
```

This adds the `BlueNoise` struct which provides true blue noise dithering
using a precomputed 256×256×4 table.

**This increases binary size by ~5M!**

```rust
#[cfg(feature = "blue-noise")]
use dithereens::{BlueNoise, simple_dither_slice_2d};

let width = 256;
let mut pixels: Vec<f32> = vec![0.5; width * width];

let blue_noise = BlueNoise::new(42);
simple_dither_slice_2d::<1, 0, _, _>(
    &mut pixels,
    width,
    255.0,
    &blue_noise,
);
```

### Float-to-Float Dithering

Dither when converting between floating-point types of different
precisions to reduce quantization artifacts like banding in smooth
gradients.

#### Supported Conversions

- **f64 → f32**: Always available.
- **f32 → f16**: Requires `nightly_f16` feature and nightly Rust.
- **f64 → f16**: Requires `nightly_f16` feature and nightly Rust.

#### Use Cases

Float-to-float dithering is particularly useful for:
- Converting HDR sky gradients from f32 to f16.
- Reducing banding in smooth color transitions.
- Maintaining visual quality when downsampling precision.
- Processing high-precision data for display or storage.

#### Example: HDR Gradient Conversion

```rust
use dithereens::dither_float_slice;

// Smooth gradient in f64.
let gradient: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.001).collect();

// Convert to f32 with dithering to preserve smoothness.
let dithered: Vec<f32> = dither_float_slice(&gradient, 42);

// Without dithering (simple cast) would show more banding.
```

#### Example: Image Conversion with 2D Methods

```rust
use dithereens::{InterleavedGradientNoise, dither_float_slice_2d};

let width = 256;
let image_f32: Vec<f32> = vec![1.5; width * width];

// Use 2D dithering for spatially-aware noise patterns.
let method = InterleavedGradientNoise::new(42);
let image_f16: Vec<f16> = dither_float_slice_2d(&image_f32, width, &method);
```

#### Available Functions

**Single values:**
- [`dither_float()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_float.html) -- Default hash method.
- [`dither_float_with()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_float_with.html) -- Custom 1D method.
- [`dither_float_2d()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_float_2d.html) -- Custom 2D method.

**Slices:**
- [`dither_float_slice()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_float_slice.html) -- 1D processing.
- [`dither_float_slice_with()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_float_slice_with.html) -- 1D with custom method.
- [`dither_float_slice_2d()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_float_slice_2d.html) -- 2D processing.

**Iterators:**
- [`dither_float_iter()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_float_iter.html) -- From iterator.
- [`dither_float_iter_with()`](https://docs.rs/dithereens/latest/dithereens/fn.dither_float_iter_with.html) -- With custom method.

**Trait methods:**
All [`LinearRng`](https://docs.rs/dithereens/latest/dithereens/trait.LinearRng.html) and [`SpatialRng`](https://docs.rs/dithereens/latest/dithereens/trait.SpatialRng.html) implementations provide
`dither_float*` methods.

See `examples/float_precision_dither.rs` for complete examples.

<!-- cargo-rdme end -->

## License

Apache-2.0 OR BSD-3-Clause OR MIT OR Zlib at your discretion.
