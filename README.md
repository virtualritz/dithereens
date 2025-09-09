# `dithereens`

![Before/after dithering](before_after_dither.png)
_Before (top) and after (bottom) dithering a gradient (uses `simple_dither()`, i.e. defaults)._

<!-- cargo-rdme start -->

Functions and traits for quantizing values with deterministic hash-based error-diffusion.

Quantizing from `f64`/`f32`/`f16` to `u32`/`u16`/`u8` without dithering
creates banding. This crate provides deterministic hash-based dithering to
reduce quantization artifacts.

## Overview

- **Deterministic**: Same input with same seed always produces same output.
- **Multiple dithering methods**: Hash, R2, GoldenRatio for 1D; IGN, SpatialHash, BlueNoise for 2D.
- **Single values**: [`dither()`], [`simple_dither()`].
- **Iterator processing**: [`dither_iter()`], [`simple_dither_iter()`].
- **In-place operations**: [`dither_slice()`], [`simple_dither_slice()`].
- **2D dithering**: [`dither_slice_2d()`], [`simple_dither_slice_2d()`] for images.
- **Custom methods**: Use specific dithering algorithms via [`*_with_method()`] functions.
- **`no_std` support**: Works in embedded environments.
- **Generic types**: `f32`, `f64`, `f16` (with `nightly_f16` feature), or
  any type implementing [`DitherFloat`].
- **Blue noise**: High-quality blue noise dithering (with `blue_noise` feature).

## Quick Start

```rust
use dithereens::simple_dither;

let value: f32 = 0.5;

// Dither `value` to `127u8` or `128u8` deterministically.
// The same index and seed will always produce the same result.
let dithered_value: u8 =
    simple_dither(value, 255.0, 0, 42).clamp(0.0, 255.0) as u8;

assert!(dithered_value == 127 || 128 == dithered_value);
```

## Dithering Methods

### 1D Methods (for sequential data)
- **Hash** (default): Fast hash-based dithering
- **R2**: Low-discrepancy sequence using the R2 sequence
- **GoldenRatio**: Golden ratio-based sequence

### 2D Methods (for images)
- **InterleavedGradientNoise (IGN)**: Fast, good quality for real-time graphics
- **SpatialHash**: Spatial hash function for blue noise-like properties
- **BlueNoiseApprox**: Approximation combining IGN and SpatialHash
- **BlueNoise** (requires `blue_noise` feature): True blue noise from precomputed tables

## Using Custom Methods

```rust
use dithereens::{simple_dither_with_method, Hash, R2, GoldenRatio};

let value = 0.5f32;
let seed = 42;

// Use different dithering methods
let hash_method = Hash::new(seed);
let r2_method = R2::new(seed);
let golden_method = GoldenRatio::new(seed);

let dithered_hash = simple_dither_with_method(value, 255.0, 0, &hash_method);
let dithered_r2 = simple_dither_with_method(value, 255.0, 0, &r2_method);
let dithered_golden = simple_dither_with_method(value, 255.0, 0, &golden_method);
```

## 2D Dithering for Images

```rust
use dithereens::{simple_dither_slice_2d, InterleavedGradientNoise};

// Example: dither a grayscale image
let width = 256;
let height = 256;
let mut pixels: Vec<f32> = vec![0.5; width * height];

// Use IGN for 2D dithering
let method = InterleavedGradientNoise::new(42);
simple_dither_slice_2d(&mut pixels, width, 255.0, &method);

// pixels now contains dithered values
```

## Performance Guide

Benchmarks with 10,000 values:

- **Single values**: [`dither()`], [`simple_dither()`].
- **In-place slice operations**: [`dither_slice()`],
  [`simple_dither_slice()`] (~5.6x faster than iterator methods)
- **Iterator chains**: [`dither_iter()`], [`simple_dither_iter()`], or
  [`DitherIteratorExt`] adapters (allocation overhead)

## Parallel Processing

Via `rayon` (enabled by default). With `rayon` enabled, `_iter` and
`_slice` functions use parallel processing automatically for better performance
on large datasets.

## `no_std` Support

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

## Native `f16` Support

Enable the `nightly_f16` feature to use native `f16` types (requires nightly
Rust):

```toml
[dependencies]
dithereens = { version = "0.3", features = ["nightly_f16"] }
```

## Blue Noise Support

Enable the `blue_noise` feature for high-quality blue noise dithering:

```toml
[dependencies]
dithereens = { version = "0.3", features = ["blue_noise"] }
```

This adds the `BlueNoise` struct which provides true blue noise dithering
using a precomputed 256×256×4 table. Note: This increases binary size by ~5MB.

```rust
#[cfg(feature = "blue_noise")]
use dithereens::{simple_dither_slice_2d, BlueNoise};

let width = 256;
let mut pixels: Vec<f32> = vec![0.5; width * width];

let blue_noise = BlueNoise::new(42);
simple_dither_slice_2d(&mut pixels, width, 255.0, &blue_noise);
```

<!-- cargo-rdme end -->

## License

Apache-2.0 OR BSD-3-Clause OR MIT OR Zlib at your discretion.
