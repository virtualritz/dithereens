# `dithereens`

![Before/after dithering](before_after_dither.png)
_Before (top) and after (bottom) dithering a gradient (uses `simple_dither()`, i.e. defaults)._

<!-- cargo-rdme start -->

Functions and traits for quantizing values with error-diffusion.

Quantizing from `f32`/`f16` to `u16`/`u8` without dithering leads to.
banding. This crate provides dithering to reduce quantization artifacts.

## Overview

- **Single values**: `dither()`, `simple_dither()`.
- **Iterator processing**: `dither_iter()`, `simple_dither_iter()`.
- **In-place operations**: `dither_slice()`, `simple_dither_slice()`.
- **Iterator adapters**: `DitherIteratorExt` for method chaining.
- **Trait-based API**: `Dither`, `SimpleDither` traits.
- **no_std support**: Works in embedded environments.
- **Generic types**: `f32`, `f64`, or any `DitherFloat` implementation.

## Quick Start

```rust
let mut rng = rand::thread_rng();

let value: f32 = 0.5;

// Dither `value` to `127u8` or `128u8`, with a probability of 50%.
// Note that we still clamp the value since it could be outside.
// the target type's range.
let dithered_value: u8 =
    simple_dither(value, 255.0, &mut rng).clamp(0.0, 255.0) as u8;

assert!(dithered_value == 127 || 128 == dithered_value);
```

## Iterator Adapters

Use `DitherIteratorExt` for ergonomic method chaining:

```rust
let mut rng = SmallRng::seed_from_u64(42);
let pixel_values = vec!0.2f32, 0.5, 0.8, 0.1, 0.9;

let result: Vec<f32> = pixel_values
    .iter()
    .copied()
    // +3/4 EV exposure.
    .map(|pixel| pixel * 2.0f32.powf(3.0 / 4.0))
    // Dither.
    .simple_dither(255.0, &mut rng);
```

## Performance Guide

Based on benchmarks with 10,000 values:

- **Single values**: `dither()`, `simple_dither()`.
- **In-place slice operations**: `dither_slice()`,
  `simple_dither_slice()` (~5.6x faster than iterator methods)
- **Iterator chains**: `dither_iter()`, `simple_dither_iter()`, or
  `DitherIteratorExt` adapters (allocation overhead)

## Parallel Processing

Via `rayon` -- enabled by default. With `rayon` enabled, `_iter` and `_slice` postfixed functions use parallel processing. The passed `RNG` must implement `Rng + Send + Clone`.

## `no_std` Support

This crate supports `no_std` environments. The `libm` crate can be used to
pull in a possibly faster, native `round()` implementation. Otherwise a
manual implementation is used in `no_std` environments.

```toml
[dependencies]
# `no_std`
dithereens = { version = "0.1", default-features = false }
# Optional: uses `libm`'s `round()` function instead of a manual implementation for `no_std`.
dithereens = { version = "0.1", default-features = false, features = ["libm"] }
```

<!-- cargo-rdme end -->

## License

Apache-2.0 OR BSD-3-Clause OR MIT OR Zlib at your discretion.
