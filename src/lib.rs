//! Functions and traits for quantizing values with deterministic hash-based
//! error-diffusion.
//!
//! Quantizing from `f64`/`f32`/`f16` to `u32`/`u16`/`u8` without dithering
//! creates banding. This crate provides deterministic hash-based dithering to
//! reduce quantization artifacts.
//!
//! ## Overview
//!
//! - **Deterministic**: Same input with same seed always produces same output.
//! - **Multiple dithering methods**: Hash, R2, GoldenRatio for 1D; IGN,
//!   SpatialHash, BlueNoise for 2D.
//! - **Single values**: [`dither()`], [`simple_dither()`].
//! - **Iterator processing**: [`dither_iter()`], [`simple_dither_iter()`].
//! - **In-place operations**: [`dither_slice()`], [`simple_dither_slice()`].
//! - **Image support**: Both 1D methods (processing as flat array) and 2D
//!   methods (using coordinates).
//! - **Custom methods**: Use specific dithering algorithms via
//!   `*_with_method()` functions.
//! - **`no_std` support**: Works in embedded environments.
//! - **Generic types**: `f32`, `f64`, `f16` (with `nightly_f16` feature), or
//!   any type implementing [`DitherFloat`].
//! - **Blue noise**: High-quality blue noise dithering (with `blue-noise`
//!   feature).
//!
//! ## Quick Start
//!
//! ```rust
//! use dithereens::simple_dither;
//!
//! let value: f32 = 0.5;
//!
//! // Dither `value` to `127u8` or `128u8` deterministically.
//! // The same index and seed will always produce the same result.
//! let dithered_value: u8 =
//!     simple_dither(value, 255.0, 0, 42).clamp(0.0, 255.0) as u8;
//!
//! assert!(dithered_value == 127 || 128 == dithered_value);
//! ```
//!
//! ## Dithering Methods
//!
//! ### 1D Methods (for sequential data and images as flat arrays)
//! - **Hash** (default): Fast hash-based dithering, good general-purpose
//!   quality.
//! - **R2**: Low-discrepancy sequence using the R2 sequence.
//! - **GoldenRatio**: Golden ratio-based sequence.
//!
//! 1D methods have been used successfully for image dithering for years by
//! processing images as flat arrays. They work well when you don't need
//! spatial correlation between pixels.
//!
//! ### 2D Methods (for images using spatial coordinates)
//! - **InterleavedGradientNoise (IGN)**: Fast, good quality for real-time
//!   graphics.
//! - **SpatialHash**: Spatial hash function for blue noise-like properties.
//! - **BlueNoiseApprox**: Approximation combining IGN and SpatialHash.
//! - **BlueNoise** (requires `blue-noise` feature): True blue noise from
//!   precomputed tables.
//!
//! 2D methods use pixel coordinates to create spatially-aware dithering
//! patterns, which can produce more visually pleasing results for images.
//!
//! ## Using Custom Methods
//!
//! ```rust
//! use dithereens::{GoldenRatio, Hash, R2, simple_dither_with};
//!
//! let value = 0.5f32;
//! let seed = 42;
//!
//! // Use different dithering methods.
//! let hash_method = Hash::new(seed);
//! let r2_method = R2::new(seed);
//! let golden_method = GoldenRatio::new(seed);
//!
//! let dithered_hash = simple_dither_with(value, 255.0, 0, &hash_method);
//! let dithered_r2 = simple_dither_with(value, 255.0, 0, &r2_method);
//! let dithered_golden = simple_dither_with(value, 255.0, 0, &golden_method);
//! ```
//!
//! ## Dynamic Method Selection
//!
//! The [`LinearDither`] and [`SpatialDither`] enums provide zero-cost dynamic
//! dispatch for runtime method selection:
//!
//! ```rust
//! use dithereens::{Hash, LinearDither, LinearRng, R2};
//!
//! // Store different methods in a collection.
//! let methods: Vec<LinearDither> = vec![
//!     LinearDither::Hash(Hash::new(1)),
//!     LinearDither::R2(R2::new(2)),
//! ];
//!
//! // All methods implement LinearRng through enum_dispatch.
//! for method in &methods {
//!     let noise = method.compute(100);
//!     // Use the noise value...
//! }
//! ```
//!
//! This is useful when:
//! - Selecting dithering methods at runtime based on configuration.
//! - Storing heterogeneous collections of methods.
//! - Implementing plugins or extensible systems.
//!
//! ## Image Dithering with 1D Methods
//!
//! 1D methods work great for images when processed as flat arrays:
//!
//! ```rust
//! use dithereens::{Hash, simple_dither_slice};
//!
//! // Example: dither a grayscale image.
//! let width = 256;
//! let height = 256;
//! let mut pixels: Vec<f32> = vec![0.5; width * height];
//!
//! // Process entire image as flat array with 1D dithering.
//! simple_dither_slice(&mut pixels, 255.0, 42);
//!
//! // pixels now contains dithered values.
//! ```
//!
//! ## 2D Dithering for Images
//!
//! 2D methods use spatial coordinates for better visual results:
//!
//! ```rust
//! use dithereens::{InterleavedGradientNoise, simple_dither_slice_2d};
//!
//! // Example: dither a grayscale image.
//! let width = 256;
//! let height = 256;
//! let mut pixels: Vec<f32> = vec![0.5; width * height];
//!
//! // Use IGN for 2D dithering (1 channel, correlated noise).
//! let method = InterleavedGradientNoise::new(42);
//! simple_dither_slice_2d::<1, 0, _, _>(&mut pixels, width, 255.0, &method);
//!
//! // pixels now contains dithered values.
//! ```
//!
//! ## Multi-Channel Image Dithering
//!
//! The 2D dithering functions support multi-channel images (RGB, RGBA) with
//! const-generic parameters for efficient processing:
//!
//! ```rust
//! use dithereens::{InterleavedGradientNoise, simple_dither_slice_2d};
//!
//! let width = 512;
//! let height = 512;
//! let method = InterleavedGradientNoise::new(42);
//!
//! // RGB image with correlated noise (same noise pattern across RGB).
//! // This is 3× faster than processing each channel separately.
//! let mut rgb_data: Vec<f32> = vec![0.5; width * height * 3];
//! simple_dither_slice_2d::<3, 0, _, _>(&mut rgb_data, width, 255.0, &method);
//!
//! // RGB image with uncorrelated noise (different pattern per channel).
//! // Provides more randomness but requires computing noise per channel.
//! let mut rgb_data2: Vec<f32> = vec![0.5; width * height * 3];
//! simple_dither_slice_2d::<3, 1, _, _>(&mut rgb_data2, width, 255.0, &method);
//!
//! // RGBA image with correlated noise.
//! let mut rgba_data: Vec<f32> = vec![0.5; width * height * 4];
//! simple_dither_slice_2d::<4, 0, _, _>(&mut rgba_data, width, 255.0, &method);
//! ```
//!
//! **Type parameters:**
//! - `CHANNELS`: Number of channels per pixel (1 = grayscale, 3 = RGB, 4 =
//!   RGBA).
//! - `SEED_OFFSET`: Per-channel noise correlation.
//!   - `0` = Correlated (same noise for all channels, fastest).
//!   - `>0` = Uncorrelated (different noise per channel, more random).
//!
//! ## Performance Guide
//!
//! Benchmarks with 10,000 values:
//!
//! - **Single values**: [`dither()`], [`simple_dither()`].
//! - **In-place slice operations**: [`dither_slice()`],
//!   [`simple_dither_slice()`] (>5× faster than iterator methods).
//! - **Iterator chains**: [`dither_iter()`], [`simple_dither_iter()`], or
//!   [`DitherIteratorExt`] adapters (allocation overhead).
//!
//! ## Parallel Processing
//!
//! Via `rayon` (enabled by default). With `rayon` enabled, `_iter` and
//! `_slice` functions use parallel processing automatically for better
//! performance on large datasets.
//!
//! ## `no_std` Support
//!
//! This crate supports `no_std` environments. The `libm` crate provides a
//! native `round()` implementation. Without `libm`, a manual implementation is
//! used.
//!
//! ```toml
//! [dependencies]
//! # `no_std`
//! dithereens = { version = "0.3", default-features = false }
//! ```
//!
//! ```toml
//! [dependencies]
//! # Optional: uses `libm`'s `round()` function instead of a manual
//! # implementation for `no_std`.
//! dithereens = {
//!    version = "0.3",
//!    default-features = false,
//!    features = ["libm"]
//! }
//! ```
//!
//! ## Native `f16` Support
//!
//! Enable the `nightly_f16` feature to use native `f16` types (requires nightly
//! Rust):
//!
//! ```toml
//! [dependencies]
//! dithereens = { version = "0.3", features = ["nightly_f16"] }
//! ```
//!
//! ## Blue Noise Support
//!
//! Enable the `blue-noise` feature for high-quality blue noise dithering:
//!
//! ```toml
//! [dependencies]
//! dithereens = { version = "0.3", features = ["blue-noise"] }
//! ```
//!
//! This adds the `BlueNoise` struct which provides true blue noise dithering
//! using a precomputed 256×256×4 table.
//!
//! **This increases binary size by ~5M!**
//!
//! ```rust
//! #[cfg(feature = "blue-noise")]
//! use dithereens::{BlueNoise, simple_dither_slice_2d};
//!
//! # #[cfg(feature = "blue-noise")]
//! # {
//! let width = 256;
//! let mut pixels: Vec<f32> = vec![0.5; width * width];
//!
//! let blue_noise = BlueNoise::new(42);
//! simple_dither_slice_2d::<1, 0, _, _>(
//!     &mut pixels,
//!     width,
//!     255.0,
//!     &blue_noise,
//! );
//! # }
//! ```
//!
//! ## Float-to-Float Dithering
//!
//! Dither when converting between floating-point types of different
//! precisions to reduce quantization artifacts like banding in smooth
//! gradients.
//!
//! ### Supported Conversions
//!
//! - **f64 → f32**: Always available.
//! - **f32 → f16**: Requires `nightly_f16` feature and nightly Rust.
//! - **f64 → f16**: Requires `nightly_f16` feature and nightly Rust.
//!
//! ### Use Cases
//!
//! Float-to-float dithering is particularly useful for:
//! - Converting HDR sky gradients from f32 to f16.
//! - Reducing banding in smooth color transitions.
//! - Maintaining visual quality when downsampling precision.
//! - Processing high-precision data for display or storage.
//!
//! ### Example: HDR Gradient Conversion
//!
//! ```rust
//! use dithereens::dither_float_slice;
//!
//! // Smooth gradient in f64.
//! let gradient: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.001).collect();
//!
//! // Convert to f32 with dithering to preserve smoothness.
//! let dithered: Vec<f32> = dither_float_slice(&gradient, 42);
//!
//! // Without dithering (simple cast) would show more banding.
//! ```
//!
//! ### Example: Image Conversion with 2D Methods
//!
//! ```rust
//! # #[cfg(feature = "nightly_f16")]
//! # {
//! use dithereens::{InterleavedGradientNoise, dither_float_slice_2d};
//!
//! let width = 256;
//! let image_f32: Vec<f32> = vec![1.5; width * width];
//!
//! // Use 2D dithering for spatially-aware noise patterns.
//! let method = InterleavedGradientNoise::new(42);
//! let image_f16: Vec<f16> = dither_float_slice_2d(&image_f32, width, &method);
//! # }
//! ```
//!
//! ### Available Functions
//!
//! **Single values:**
//! - [`dither_float()`] -- Default hash method.
//! - [`dither_float_with()`] -- Custom 1D method.
//! - [`dither_float_2d()`] -- Custom 2D method.
//!
//! **Slices:**
//! - [`dither_float_slice()`] -- 1D processing.
//! - [`dither_float_slice_with()`] -- 1D with custom method.
//! - [`dither_float_slice_2d()`] -- 2D processing.
//!
//! **Iterators:**
//! - [`dither_float_iter()`] -- From iterator.
//! - [`dither_float_iter_with()`] -- With custom method.
//!
//! **Trait methods:**
//! All [`LinearRng`] and [`SpatialRng`] implementations provide
//! `dither_float*` methods.
//!
//! See `examples/float_precision_dither.rs` for complete examples.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly_f16", feature(f16))]

#[cfg(feature = "blue-noise")]
mod blue_noise;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "nightly_f16"))]
use common_traits::{CastableFrom, Number};

#[cfg(feature = "nightly_f16")]
use common_traits_f16::{CastableFrom, Number};
use core::{
    cmp::PartialOrd,
    ops::{Add, Mul, Neg, Sub},
};
use enum_dispatch::enum_dispatch;

/// Trait for linear (1D) random number generators.
///
/// These generators produce deterministic random values based on a single
/// index, making them ideal for sequential processing and 1D dithering
/// applications.
///
/// # When to Use LinearRng
///
/// Use [`LinearRng`] implementations when:
/// - Processing data sequentially (arrays, streams).
/// - Dithering images as flat pixel arrays.
/// - You need consistent results with a given seed.
/// - Memory efficiency matters (no lookup tables).
///
/// # Available Implementations
///
/// - [`struct@Hash`] -- Fast hash-based RNG. Good general-purpose quality with
///   uniform distribution.
/// - [`R2`] -- Low-discrepancy sequence using the R2 recurrence. Better spatial
///   distribution than random.
/// - [`GoldenRatio`] -- Golden ratio sequence. Optimal 1D coverage with minimal
///   clustering.
#[enum_dispatch]
pub trait LinearRng: Sized + Send + Sync {
    /// Compute dither offset for a given index.
    /// Returns a value in range [-1, 1] to be scaled by amplitude.
    fn compute(&self, index: u32) -> f32;

    /// Dither a single value.
    #[inline]
    fn dither<T>(
        &self,
        value: T,
        min: T,
        one: T,
        dither_amplitude: T,
        index: u32,
    ) -> T
    where
        T: DitherFloat,
        Self: Sized,
    {
        dither_with(value, min, one, dither_amplitude, index, self)
    }

    /// Simple dither for a single value.
    #[inline]
    fn simple_dither<T>(&self, value: T, one: T, index: u32) -> T
    where
        T: DitherFloat + Number + CastableFrom<f32>,
        Self: Sized,
    {
        simple_dither_with(value, one, index, self)
    }

    /// Dither values in a slice.
    #[cfg(not(feature = "rayon"))]
    fn dither_slice<T>(
        &self,
        values: &mut [T],
        min: T,
        one: T,
        dither_amplitude: T,
    ) where
        T: DitherFloat,
        Self: Sized,
    {
        dither_slice_with(values, min, one, dither_amplitude, self)
    }

    /// Dither values in a slice (parallel version).
    #[cfg(feature = "rayon")]
    fn dither_slice<T>(
        &self,
        values: &mut [T],
        min: T,
        one: T,
        dither_amplitude: T,
    ) where
        T: DitherFloat + Send + Sync,
        Self: Sized,
    {
        dither_slice_with(values, min, one, dither_amplitude, self)
    }

    /// Simple dither for values in a slice.
    #[cfg(not(feature = "rayon"))]
    fn simple_dither_slice<T>(&self, values: &mut [T], one: T)
    where
        T: DitherFloat + Number + CastableFrom<f32>,
        Self: Sized,
    {
        simple_dither_slice_with(values, one, self)
    }

    /// Simple dither for values in a slice (parallel version).
    #[cfg(feature = "rayon")]
    fn simple_dither_slice<T>(&self, values: &mut [T], one: T)
    where
        T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
        Self: Sized,
    {
        simple_dither_slice_with(values, one, self)
    }

    /// Dither values from an iterator.
    #[cfg(not(feature = "rayon"))]
    fn dither_iter<T, I>(
        &self,
        values: I,
        min: T,
        one: T,
        dither_amplitude: T,
    ) -> Vec<T>
    where
        T: DitherFloat,
        I: IntoIterator<Item = T>,
        Self: Sized,
    {
        dither_iter_with(values, min, one, dither_amplitude, self)
    }

    /// Dither values from an iterator (parallel version).
    #[cfg(feature = "rayon")]
    fn dither_iter<T, I>(
        &self,
        values: I,
        min: T,
        one: T,
        dither_amplitude: T,
    ) -> Vec<T>
    where
        T: DitherFloat + Send + Sync,
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        Self: Sized,
    {
        dither_iter_with(values, min, one, dither_amplitude, self)
    }

    /// Simple dither for values from an iterator.
    #[cfg(not(feature = "rayon"))]
    fn simple_dither_iter<T, I>(&self, values: I, one: T) -> Vec<T>
    where
        T: DitherFloat + Number + CastableFrom<f32>,
        I: IntoIterator<Item = T>,
        Self: Sized,
    {
        simple_dither_iter_with(values, one, self)
    }

    /// Simple dither for values from an iterator (parallel version).
    #[cfg(feature = "rayon")]
    fn simple_dither_iter<T, I>(&self, values: I, one: T) -> Vec<T>
    where
        T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        Self: Sized,
    {
        simple_dither_iter_with(values, one, self)
    }

    /// Dither a float value when converting to lower precision.
    #[inline]
    fn dither_float<Src, Dest>(&self, value: Src, index: u32) -> Dest
    where
        Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64>,
        Self: Sized,
    {
        dither_float_with(value, index, self)
    }

    /// Dither float values in a slice when converting to lower precision.
    #[cfg(not(feature = "rayon"))]
    fn dither_float_slice<Src, Dest>(&self, values: &[Src]) -> Vec<Dest>
    where
        Src: DitherFloatConversion<Dest>
            + DitherFloat
            + CastableFrom<f64>
            + Copy,
        Self: Sized,
    {
        dither_float_slice_with(values, self)
    }

    /// Dither float values in a slice when converting to lower precision
    /// (parallel version).
    #[cfg(feature = "rayon")]
    fn dither_float_slice<Src, Dest>(&self, values: &[Src]) -> Vec<Dest>
    where
        Src: DitherFloatConversion<Dest>
            + DitherFloat
            + CastableFrom<f64>
            + Copy
            + Send
            + Sync,
        Dest: Send,
        Self: Sized,
    {
        dither_float_slice_with(values, self)
    }

    /// Dither float values from an iterator when converting to lower
    /// precision.
    #[cfg(not(feature = "rayon"))]
    fn dither_float_iter<Src, Dest, I>(&self, values: I) -> Vec<Dest>
    where
        Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64>,
        I: IntoIterator<Item = Src>,
        Self: Sized,
    {
        dither_float_iter_with(values, self)
    }

    /// Dither float values from an iterator when converting to lower precision
    /// (parallel version).
    #[cfg(feature = "rayon")]
    fn dither_float_iter<Src, Dest, I>(&self, values: I) -> Vec<Dest>
    where
        Src: DitherFloatConversion<Dest>
            + DitherFloat
            + CastableFrom<f64>
            + Send
            + Sync,
        Dest: Send,
        I: IntoIterator<Item = Src>,
        I::IntoIter: Send,
        Self: Sized,
    {
        dither_float_iter_with(values, self)
    }
}

/// Trait for spatial random number generators.
///
/// These generators produce deterministic random values based on 2D
/// coordinates, making them ideal for parallel processing and applications like
/// dithering.
///
/// # When to Use SpatialRng
///
/// Use [`SpatialRng`] implementations when:
/// - Processing 2D data (images, textures).
/// - You need spatially-aware noise patterns.
/// - Parallel processing is important (each pixel independent).
/// - Visual quality matters more than speed.
///
/// # Available Implementations
///
/// - [`InterleavedGradientNoise`] -- Fast IGN algorithm from Jorge Jimenez.
///   Real-time graphics quality.
/// - [`SpatialHash`] -- Spatial hash function. Blue noise-like properties with
///   good performance.
/// - [`BlueNoiseApprox`] -- Combines IGN and [`SpatialHash`]. Approximates blue
///   noise characteristics.
/// - `BlueNoise` (requires `blue-noise` feature) -- True blue noise from
///   precomputed tables. Highest quality.
#[enum_dispatch]
pub trait SpatialRng: Sized + Send + Sync {
    /// Compute a deterministic value for given 2D coordinates.
    /// Returns a value in range [-1, 1].
    fn compute(&self, x: u32, y: u32) -> f32;

    /// Dither a single value using 2D coordinates.
    #[inline]
    fn dither_2d<T>(
        &self,
        value: T,
        min: T,
        one: T,
        dither_amplitude: T,
        x: u32,
        y: u32,
    ) -> T
    where
        T: DitherFloat,
        Self: Sized,
    {
        dither_2d(value, min, one, dither_amplitude, x, y, self)
    }

    /// Simple dither for a single value using 2D coordinates.
    #[inline]
    fn simple_dither_2d<T>(&self, value: T, one: T, x: u32, y: u32) -> T
    where
        T: DitherFloat + Number + CastableFrom<f32>,
        Self: Sized,
    {
        simple_dither_2d(value, one, x, y, self)
    }

    /// Dither a 2D image stored as a flat slice with multi-channel support.
    #[cfg(not(feature = "rayon"))]
    fn dither_slice_2d<const CHANNELS: usize, const SEED_OFFSET: u32, T>(
        &self,
        values: &mut [T],
        width: usize,
        min: T,
        one: T,
        dither_amplitude: T,
    ) where
        T: DitherFloat,
        Self: Sized,
    {
        dither_slice_2d::<CHANNELS, SEED_OFFSET, T, Self>(
            values,
            width,
            min,
            one,
            dither_amplitude,
            self,
        )
    }

    /// Dither a 2D image stored as a flat slice (parallel version).
    #[cfg(feature = "rayon")]
    fn dither_slice_2d<const CHANNELS: usize, const SEED_OFFSET: u32, T>(
        &self,
        values: &mut [T],
        width: usize,
        min: T,
        one: T,
        dither_amplitude: T,
    ) where
        T: DitherFloat + Send + Sync,
        Self: Sized,
    {
        dither_slice_2d::<CHANNELS, SEED_OFFSET, T, Self>(
            values,
            width,
            min,
            one,
            dither_amplitude,
            self,
        )
    }

    /// Simple dither for a 2D image stored as a flat slice with multi-channel
    /// support.
    #[cfg(not(feature = "rayon"))]
    fn simple_dither_slice_2d<
        const CHANNELS: usize,
        const SEED_OFFSET: u32,
        T,
    >(
        &self,
        values: &mut [T],
        width: usize,
        one: T,
    ) where
        T: DitherFloat + Number + CastableFrom<f32>,
        Self: Sized,
    {
        simple_dither_slice_2d::<CHANNELS, SEED_OFFSET, T, Self>(
            values, width, one, self,
        )
    }

    /// Simple dither for a 2D image stored as a flat slice (parallel version).
    #[cfg(feature = "rayon")]
    fn simple_dither_slice_2d<
        const CHANNELS: usize,
        const SEED_OFFSET: u32,
        T,
    >(
        &self,
        values: &mut [T],
        width: usize,
        one: T,
    ) where
        T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
        Self: Sized,
    {
        simple_dither_slice_2d::<CHANNELS, SEED_OFFSET, T, Self>(
            values, width, one, self,
        )
    }

    /// Dither a float value when converting to lower precision using 2D
    /// coordinates.
    #[inline]
    fn dither_float_2d<Src, Dest>(&self, value: Src, x: u32, y: u32) -> Dest
    where
        Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64>,
        Self: Sized,
    {
        dither_float_2d(value, x, y, self)
    }

    /// Dither float values in a 2D image slice when converting to lower
    /// precision.
    #[cfg(not(feature = "rayon"))]
    fn dither_float_slice_2d<Src, Dest>(
        &self,
        values: &[Src],
        width: usize,
    ) -> Vec<Dest>
    where
        Src: DitherFloatConversion<Dest>
            + DitherFloat
            + CastableFrom<f64>
            + Copy,
        Self: Sized,
    {
        dither_float_slice_2d(values, width, self)
    }

    /// Dither float values in a 2D image slice when converting to lower
    /// precision (parallel version).
    #[cfg(feature = "rayon")]
    fn dither_float_slice_2d<Src, Dest>(
        &self,
        values: &[Src],
        width: usize,
    ) -> Vec<Dest>
    where
        Src: DitherFloatConversion<Dest>
            + DitherFloat
            + CastableFrom<f64>
            + Copy
            + Send
            + Sync,
        Dest: Send,
        Self: Sized,
    {
        dither_float_slice_2d(values, width, self)
    }
}

mod linear;
mod spatial;

pub use linear::*;
pub use spatial::*;

/// Minimal trait requirements for dithering.
pub trait DitherFloat:
    Copy
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Number
    + CastableFrom<f64>
{
    fn round(self) -> Self;
}

impl DitherFloat for f32 {
    #[cfg(feature = "std")]
    fn round(self) -> Self {
        self.round()
    }

    #[cfg(all(not(feature = "std"), feature = "libm"))]
    fn round(self) -> Self {
        libm::roundf(self)
    }

    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    fn round(self) -> Self {
        // Basic rounding without libm - less precise but works
        if self >= 0.0 {
            (self + 0.5) as i32 as f32
        } else {
            (self - 0.5) as i32 as f32
        }
    }
}

impl DitherFloat for f64 {
    #[cfg(feature = "std")]
    fn round(self) -> Self {
        self.round()
    }

    #[cfg(all(not(feature = "std"), feature = "libm"))]
    fn round(self) -> Self {
        libm::round(self)
    }

    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    fn round(self) -> Self {
        // Basic rounding without libm - less precise but works
        if self >= 0.0 {
            (self + 0.5) as i64 as f64
        } else {
            (self - 0.5) as i64 as f64
        }
    }
}

#[cfg(feature = "nightly_f16")]
impl DitherFloat for f16 {
    #[cfg(feature = "std")]
    fn round(self) -> Self {
        // Convert to f32 for rounding, then back to f16
        (self as f32).round() as f16
    }

    #[cfg(all(not(feature = "std"), feature = "libm"))]
    fn round(self) -> Self {
        // Convert to f32 for rounding, then back to f16
        libm::roundf(self as f32) as f16
    }

    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    fn round(self) -> Self {
        // Basic rounding without libm - less precise but works
        let f32_val = self as f32;
        let rounded = if f32_val >= 0.0 {
            (f32_val + 0.5) as i32 as f32
        } else {
            (f32_val - 0.5) as i32 as f32
        };
        rounded as f16
    }
}

/// Trait for dithered float-to-float precision conversions.
///
/// This trait enables dithering when converting between floating-point types
/// of different precisions to reduce quantization artifacts like banding in
/// smooth gradients.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "nightly_f16")]
/// # {
/// use dithereens::dither_float;
///
/// let value: f32 = 1.234567;
/// let dithered: f16 = dither_float(value, 0, 42);
/// # }
/// ```
pub trait DitherFloatConversion<Dest>: Sized {
    /// Compute the ULP (Unit in the Last Place) for the destination type at
    /// this value's magnitude.
    ///
    /// The ULP represents the quantization step size -- the spacing between
    /// representable values at the current magnitude.
    fn compute_target_ulp(self) -> Self;

    /// Cast to the destination type.
    fn cast_to_dest(self) -> Dest;
}

/// Implementation for f32 to f16 conversion (requires `nightly_f16` feature).
///
/// f16 has 10 mantissa bits vs f32's 23 bits, losing 13 bits of precision.
#[cfg(feature = "nightly_f16")]
impl DitherFloatConversion<f16> for f32 {
    fn compute_target_ulp(self) -> Self {
        // Handle special cases.
        if !self.is_finite() {
            return 1.0; // Doesn't matter, won't be used.
        }

        let abs_val = self.abs();

        // f16 range is approximately ±65504.
        // Values outside this range will become infinity.
        const F16_MAX: f32 = 65504.0;
        if abs_val >= F16_MAX {
            return 1.0; // Will saturate to infinity, dithering not helpful.
        }

        // f16 subnormals have fixed ULP.
        const F16_MIN_NORMAL: f32 = 6.103_515_6e-5; // 2^-14
        if abs_val < F16_MIN_NORMAL {
            // Subnormal range has fixed ULP of 2^-24.
            return 5.960_464_5e-8; // 2^-24
        }

        // Normal range: extract exponent and compute ULP.
        // f16 loses 13 mantissa bits compared to f32.
        let bits = self.to_bits();
        let exponent = ((bits >> 23) & 0xFF) as i32 - 127;

        // ULP for f16 at this exponent = 2^(exponent - 10)
        // (f16 has 10 mantissa bits)
        2.0_f32.powi(exponent - 10)
    }

    #[inline(always)]
    fn cast_to_dest(self) -> f16 {
        self as f16
    }
}

/// Implementation for f64 to f16 conversion (requires `nightly_f16` feature).
///
/// f16 has 10 mantissa bits vs f64's 52 bits, losing 42 bits of precision.
#[cfg(feature = "nightly_f16")]
impl DitherFloatConversion<f16> for f64 {
    fn compute_target_ulp(self) -> Self {
        // Handle special cases.
        if !self.is_finite() {
            return 1.0;
        }

        let abs_val = self.abs();

        // f16 range is approximately ±65504.
        const F16_MAX: f64 = 65504.0;
        if abs_val >= F16_MAX {
            return 1.0;
        }

        // f16 subnormals have fixed ULP.
        const F16_MIN_NORMAL: f64 = 6.103515625e-5; // 2^-14
        if abs_val < F16_MIN_NORMAL {
            return 5.960464477539063e-8; // 2^-24
        }

        // Normal range: extract exponent and compute ULP.
        let bits = self.to_bits();
        let exponent = ((bits >> 52) & 0x7FF) as i32 - 1023;

        // ULP for f16 at this exponent = 2^(exponent - 10)
        2.0_f64.powi(exponent - 10)
    }

    #[inline(always)]
    fn cast_to_dest(self) -> f16 {
        self as f16
    }
}

/// Implementation for f64 to f32 conversion.
///
/// f32 has 23 mantissa bits vs f64's 52 bits, losing 29 bits of precision.
impl DitherFloatConversion<f32> for f64 {
    fn compute_target_ulp(self) -> Self {
        // Handle special cases.
        if !self.is_finite() {
            return 1.0;
        }

        let abs_val = self.abs();

        // f32 range is approximately ±3.4e38.
        const F32_MAX: f64 = 3.4028234663852886e38;
        if abs_val >= F32_MAX {
            return 1.0;
        }

        // f32 subnormals have fixed ULP.
        const F32_MIN_NORMAL: f64 = 1.1754943508222875e-38; // 2^-126
        if abs_val < F32_MIN_NORMAL {
            return 1.401298464324817e-45; // 2^-149
        }

        // Normal range: extract exponent and compute ULP.
        let bits = self.to_bits();
        let exponent = ((bits >> 52) & 0x7FF) as i32 - 1023;

        // ULP for f32 at this exponent = 2^(exponent - 23)
        // (f32 has 23 mantissa bits)
        2.0_f64.powi(exponent - 23)
    }

    #[inline(always)]
    fn cast_to_dest(self) -> f32 {
        self as f32
    }
}

// Note: No longer can have a static default hash since it needs a seed

/// Dither a value using the default hash method (backward compatible).
#[inline]
pub fn dither<T>(
    value: T,
    min: T,
    one: T,
    dither_amplitude: T,
    index: u32,
    seed: u32,
) -> T
where
    T: DitherFloat,
{
    let method = Hash::new(seed);
    dither_with(value, min, one, dither_amplitude, index, &method)
}

/// Dither a value using a specific method.
#[inline]
pub fn dither_with<T, M: LinearRng>(
    value: T,
    min: T,
    one: T,
    dither_amplitude: T,
    index: u32,
    method: &M,
) -> T
where
    T: DitherFloat,
{
    let dither = if dither_amplitude == T::ZERO {
        T::ZERO
    } else {
        T::cast_from(method.compute(index) as f64) * dither_amplitude
    };

    (min + value * (one - min) + dither).round()
}

/// Simple dither with default hash method (backward compatible).
#[inline]
pub fn simple_dither<T>(value: T, one: T, index: u32, seed: u32) -> T
where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    let method = Hash::new(seed);
    simple_dither_with(value, one, index, &method)
}

/// Simple dither with specific method.
#[inline]
pub fn simple_dither_with<T, M: LinearRng>(
    value: T,
    one: T,
    index: u32,
    method: &M,
) -> T
where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    dither_with(value, T::ZERO, one, T::cast_from(0.5_f32), index, method)
        .clamp(T::ZERO, one)
}

/// Dither a value using 2D coordinates.
#[inline]
pub fn dither_2d<T, M: SpatialRng>(
    value: T,
    min: T,
    one: T,
    dither_amplitude: T,
    x: u32,
    y: u32,
    method: &M,
) -> T
where
    T: DitherFloat,
{
    let dither = if dither_amplitude == T::ZERO {
        T::ZERO
    } else {
        T::cast_from(method.compute(x, y) as f64) * dither_amplitude
    };

    (min + value * (one - min) + dither).round()
}

/// Simple dither with 2D coordinates.
#[inline]
pub fn simple_dither_2d<T, M: SpatialRng>(
    value: T,
    one: T,
    x: u32,
    y: u32,
    method: &M,
) -> T
where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    dither_2d(value, T::ZERO, one, T::cast_from(0.5_f32), x, y, method)
        .clamp(T::ZERO, one)
}

/// Dither a float value when converting to lower precision using default hash
/// method.
///
/// This reduces quantization artifacts like banding in smooth gradients when
/// converting between floating-point types (e.g., f32 to f16, f64 to f32).
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "nightly_f16")]
/// # {
/// use dithereens::dither_float;
///
/// let value: f32 = 1.234567;
/// let result: f16 = dither_float(value, 0, 42);
/// # }
/// ```
///
/// ```rust
/// use dithereens::dither_float;
///
/// let value: f64 = 1.234567890123456;
/// let result: f32 = dither_float(value, 0, 42);
/// ```
#[inline]
pub fn dither_float<Src, Dest>(value: Src, index: u32, seed: u32) -> Dest
where
    Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64>,
{
    let method = Hash::new(seed);
    dither_float_with(value, index, &method)
}

/// Dither a float value when converting to lower precision using a specific
/// linear RNG method.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "nightly_f16")]
/// # {
/// use dithereens::{R2, dither_float_with};
///
/// let value: f32 = 1.234567;
/// let method = R2::new(42);
/// let result: f16 = dither_float_with(value, 0, &method);
/// # }
/// ```
#[inline]
pub fn dither_float_with<Src, Dest, M: LinearRng>(
    value: Src,
    index: u32,
    method: &M,
) -> Dest
where
    Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64>,
{
    let ulp = value.compute_target_ulp();
    let dither_noise = Src::cast_from(method.compute(index) as f64);
    let dithered = value + dither_noise * ulp * Src::cast_from(0.5);
    dithered.cast_to_dest()
}

/// Dither a float value when converting to lower precision using 2D spatial
/// RNG method.
///
/// This is ideal for images and textures where spatial coordinates provide
/// better dithering patterns.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "nightly_f16")]
/// # {
/// use dithereens::{InterleavedGradientNoise, dither_float_2d};
///
/// let value: f32 = 1.234567;
/// let method = InterleavedGradientNoise::new(42);
/// let result: f16 = dither_float_2d(value, 10, 20, &method);
/// # }
/// ```
#[inline]
pub fn dither_float_2d<Src, Dest, M: SpatialRng>(
    value: Src,
    x: u32,
    y: u32,
    method: &M,
) -> Dest
where
    Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64>,
{
    let ulp = value.compute_target_ulp();
    let dither_noise = Src::cast_from(method.compute(x, y) as f64);
    let dithered = value + dither_noise * ulp * Src::cast_from(0.5);
    dithered.cast_to_dest()
}

/// Dither values in a slice using default hash method.
#[cfg(not(feature = "rayon"))]
pub fn dither_slice<T>(
    values: &mut [T],
    min: T,
    one: T,
    dither_amplitude: T,
    seed: u32,
) where
    T: DitherFloat,
{
    let method = Hash::new(seed);
    dither_slice_with(values, min, one, dither_amplitude, &method)
}

#[cfg(feature = "rayon")]
pub fn dither_slice<T>(
    values: &mut [T],
    min: T,
    one: T,
    dither_amplitude: T,
    seed: u32,
) where
    T: DitherFloat + Send + Sync,
{
    let method = Hash::new(seed);
    dither_slice_with(values, min, one, dither_amplitude, &method)
}

/// Dither values in a slice using specific method.
#[cfg(not(feature = "rayon"))]
pub fn dither_slice_with<T, M: LinearRng>(
    values: &mut [T],
    min: T,
    one: T,
    dither_amplitude: T,
    method: &M,
) where
    T: DitherFloat,
{
    for (index, value) in values.iter_mut().enumerate() {
        *value = dither_with(
            *value,
            min,
            one,
            dither_amplitude,
            index as u32,
            method,
        );
    }
}

#[cfg(feature = "rayon")]
pub fn dither_slice_with<T, M: LinearRng>(
    values: &mut [T],
    min: T,
    one: T,
    dither_amplitude: T,
    method: &M,
) where
    T: DitherFloat + Send + Sync,
{
    use rayon::prelude::*;

    values
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            *value = dither_with(
                *value,
                min,
                one,
                dither_amplitude,
                index as u32,
                method,
            );
        });
}

/// Dither a 2D image stored as a flat slice with full control and multi-channel
/// support.
///
/// Provides complete control over dithering parameters while supporting
/// multi-channel images (e.g., RGB, RGBA) with efficient noise computation.
///
/// # Type Parameters
/// - `CHANNELS`: Number of channels per pixel (1 for grayscale, 3 for RGB, 4
///   for RGBA).
/// - `SEED_OFFSET`: Per-channel seed offset (0 = correlated noise across
///   channels, non-zero = different noise per channel).
///
/// # Parameters
/// - `values`: Flat slice of interleaved pixel data (length = width * height *
///   CHANNELS).
/// - `width`: Image width in pixels.
/// - `min`: Minimum output value.
/// - `one`: Maximum output value.
/// - `dither_amplitude`: Controls dithering strength.
/// - `method`: Spatial dithering method implementing [`SpatialRng`].
///
/// # Example
///
/// ```rust
/// use dithereens::{InterleavedGradientNoise, dither_slice_2d};
///
/// let width = 256;
/// let height = 256;
/// let mut rgb_data: Vec<f32> = vec![0.5; width * height * 3];
/// let method = InterleavedGradientNoise::new(42);
///
/// // Dither RGB with correlated noise.
/// dither_slice_2d::<3, 0, _, _>(
///     &mut rgb_data,
///     width,
///     0.0,
///     255.0,
///     0.5,
///     &method,
/// );
/// ```
#[cfg(not(feature = "rayon"))]
pub fn dither_slice_2d<const CHANNELS: usize, const SEED_OFFSET: u32, T, M>(
    values: &mut [T],
    width: usize,
    min: T,
    one: T,
    dither_amplitude: T,
    method: &M,
) where
    T: DitherFloat,
    M: SpatialRng,
{
    let height = values.len() / (width * CHANNELS);

    if SEED_OFFSET == 0 {
        // Correlated noise: compute once per pixel, apply to all channels.
        for y in 0..height {
            for x in 0..width {
                let noise = method.compute(x as u32, y as u32);
                let dither = if dither_amplitude == T::ZERO {
                    T::ZERO
                } else {
                    T::cast_from(noise as f64) * dither_amplitude
                };

                let pixel_start = (y * width + x) * CHANNELS;
                for c in 0..CHANNELS {
                    let idx = pixel_start + c;
                    values[idx] =
                        (min + values[idx] * (one - min) + dither).round();
                }
            }
        }
    } else {
        // Uncorrelated noise: compute per channel using seed offset.
        for y in 0..height {
            for x in 0..width {
                let pixel_start = (y * width + x) * CHANNELS;
                for c in 0..CHANNELS {
                    let channel_offset = (c as u32) * SEED_OFFSET;
                    let noise = method.compute(
                        (x as u32).wrapping_add(channel_offset),
                        (y as u32).wrapping_add(channel_offset),
                    );
                    let dither = if dither_amplitude == T::ZERO {
                        T::ZERO
                    } else {
                        T::cast_from(noise as f64) * dither_amplitude
                    };

                    let idx = pixel_start + c;
                    values[idx] =
                        (min + values[idx] * (one - min) + dither).round();
                }
            }
        }
    }
}

#[cfg(feature = "rayon")]
pub fn dither_slice_2d<const CHANNELS: usize, const SEED_OFFSET: u32, T, M>(
    values: &mut [T],
    width: usize,
    min: T,
    one: T,
    dither_amplitude: T,
    method: &M,
) where
    T: DitherFloat + Send + Sync,
    M: SpatialRng,
{
    use rayon::prelude::*;

    let row_size = width * CHANNELS;

    if SEED_OFFSET == 0 {
        // Correlated noise: process rows in parallel using par_chunks_mut.
        values
            .par_chunks_mut(row_size)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let noise = method.compute(x as u32, y as u32);
                    let dither = if dither_amplitude == T::ZERO {
                        T::ZERO
                    } else {
                        T::cast_from(noise as f64) * dither_amplitude
                    };

                    let pixel_start = x * CHANNELS;
                    for c in 0..CHANNELS {
                        let idx = pixel_start + c;
                        row[idx] =
                            (min + row[idx] * (one - min) + dither).round();
                    }
                }
            });
    } else {
        // Uncorrelated noise: process rows in parallel using par_chunks_mut.
        values
            .par_chunks_mut(row_size)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let pixel_start = x * CHANNELS;
                    for c in 0..CHANNELS {
                        let channel_offset = (c as u32) * SEED_OFFSET;
                        let noise = method.compute(
                            (x as u32).wrapping_add(channel_offset),
                            (y as u32).wrapping_add(channel_offset),
                        );
                        let dither = if dither_amplitude == T::ZERO {
                            T::ZERO
                        } else {
                            T::cast_from(noise as f64) * dither_amplitude
                        };

                        let idx = pixel_start + c;
                        row[idx] =
                            (min + row[idx] * (one - min) + dither).round();
                    }
                }
            });
    }
}

/// Simple dither for slices using default hash method.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_slice<T>(values: &mut [T], one: T, seed: u32)
where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    let method = Hash::new(seed);
    simple_dither_slice_with(values, one, &method)
}

#[cfg(feature = "rayon")]
pub fn simple_dither_slice<T>(values: &mut [T], one: T, seed: u32)
where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
{
    let method = Hash::new(seed);
    simple_dither_slice_with(values, one, &method)
}

/// Simple dither for slices using specific method.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_slice_with<T, M: LinearRng>(
    values: &mut [T],
    one: T,
    method: &M,
) where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    for (index, value) in values.iter_mut().enumerate() {
        *value = simple_dither_with(*value, one, index as u32, method);
    }
}

#[cfg(feature = "rayon")]
pub fn simple_dither_slice_with<T, M: LinearRng>(
    values: &mut [T],
    one: T,
    method: &M,
) where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
{
    use rayon::prelude::*;

    values
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            *value = simple_dither_with(*value, one, index as u32, method);
        });
}

/// Simple dither for 2D slices with multi-channel support.
///
/// Processes multi-channel image data (e.g., RGB, RGBA) efficiently by
/// computing noise once per pixel coordinate and optionally sharing it across
/// channels.
///
/// # Type Parameters
/// - `CHANNELS`: Number of channels per pixel (1 for grayscale, 3 for RGB, 4
///   for RGBA).
/// - `SEED_OFFSET`: Per-channel seed offset (0 = correlated noise across
///   channels, non-zero = different noise per channel).
///
/// # Parameters
/// - `values`: Flat slice of interleaved pixel data (length = width * height *
///   CHANNELS).
/// - `width`: Image width in pixels.
/// - `one`: Maximum output value (typically 255.0).
/// - `method`: Spatial dithering method implementing [`SpatialRng`].
///
/// # Example
///
/// ```rust
/// use dithereens::{InterleavedGradientNoise, simple_dither_slice_2d};
///
/// let width = 256;
/// let height = 256;
///
/// // RGB image with correlated noise (same noise for all channels).
/// let mut rgb_data: Vec<f32> = vec![0.5; width * height * 3];
/// let method = InterleavedGradientNoise::new(42);
/// simple_dither_slice_2d::<3, 0, _, _>(&mut rgb_data, width, 255.0, &method);
///
/// // RGB image with uncorrelated noise (different noise per channel).
/// let mut rgb_data2: Vec<f32> = vec![0.5; width * height * 3];
/// simple_dither_slice_2d::<3, 1, _, _>(&mut rgb_data2, width, 255.0, &method);
/// ```
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_slice_2d<
    const CHANNELS: usize,
    const SEED_OFFSET: u32,
    T,
    M,
>(
    values: &mut [T],
    width: usize,
    one: T,
    method: &M,
) where
    T: DitherFloat + Number + CastableFrom<f32>,
    M: SpatialRng,
{
    let height = values.len() / (width * CHANNELS);

    if SEED_OFFSET == 0 {
        // Correlated noise: compute once per pixel, apply to all channels.
        for y in 0..height {
            for x in 0..width {
                let noise = method.compute(x as u32, y as u32);
                let dither_offset =
                    T::cast_from(noise as f64) * T::cast_from(0.5_f32);

                let pixel_start = (y * width + x) * CHANNELS;
                for c in 0..CHANNELS {
                    let idx = pixel_start + c;
                    let scaled = values[idx] * one;
                    values[idx] =
                        (scaled + dither_offset).round().clamp(T::ZERO, one);
                }
            }
        }
    } else {
        // Uncorrelated noise: compute per channel using seed offset.
        for y in 0..height {
            for x in 0..width {
                let pixel_start = (y * width + x) * CHANNELS;
                for c in 0..CHANNELS {
                    // Compute different noise for each channel by offsetting
                    // coordinates.
                    let channel_offset = (c as u32) * SEED_OFFSET;
                    let noise = method.compute(
                        (x as u32).wrapping_add(channel_offset),
                        (y as u32).wrapping_add(channel_offset),
                    );
                    let dither_offset =
                        T::cast_from(noise as f64) * T::cast_from(0.5_f32);

                    let idx = pixel_start + c;
                    let scaled = values[idx] * one;
                    values[idx] =
                        (scaled + dither_offset).round().clamp(T::ZERO, one);
                }
            }
        }
    }
}

#[cfg(feature = "rayon")]
pub fn simple_dither_slice_2d<
    const CHANNELS: usize,
    const SEED_OFFSET: u32,
    T,
    M,
>(
    values: &mut [T],
    width: usize,
    one: T,
    method: &M,
) where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
    M: SpatialRng,
{
    use rayon::prelude::*;

    let row_size = width * CHANNELS;

    if SEED_OFFSET == 0 {
        // Correlated noise: process rows in parallel using par_chunks_mut.
        values
            .par_chunks_mut(row_size)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let noise = method.compute(x as u32, y as u32);
                    let dither_offset =
                        T::cast_from(noise as f64) * T::cast_from(0.5_f32);

                    let pixel_start = x * CHANNELS;
                    for c in 0..CHANNELS {
                        let idx = pixel_start + c;
                        let scaled = row[idx] * one;
                        row[idx] = (scaled + dither_offset)
                            .round()
                            .clamp(T::ZERO, one);
                    }
                }
            });
    } else {
        // Uncorrelated noise: process rows in parallel using par_chunks_mut.
        values
            .par_chunks_mut(row_size)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let pixel_start = x * CHANNELS;
                    for c in 0..CHANNELS {
                        let channel_offset = (c as u32) * SEED_OFFSET;
                        let noise = method.compute(
                            (x as u32).wrapping_add(channel_offset),
                            (y as u32).wrapping_add(channel_offset),
                        );
                        let dither_offset =
                            T::cast_from(noise as f64) * T::cast_from(0.5_f32);

                        let idx = pixel_start + c;
                        let scaled = row[idx] * one;
                        row[idx] = (scaled + dither_offset)
                            .round()
                            .clamp(T::ZERO, one);
                    }
                }
            });
    }
}

/// Dither values from an iterator using default hash method.
#[cfg(not(feature = "rayon"))]
pub fn dither_iter<T, I>(
    values: I,
    min: T,
    one: T,
    dither_amplitude: T,
    seed: u32,
) -> Vec<T>
where
    T: DitherFloat,
    I: IntoIterator<Item = T>,
{
    let method = Hash::new(seed);
    dither_iter_with(values, min, one, dither_amplitude, &method)
}

#[cfg(feature = "rayon")]
pub fn dither_iter<T, I>(
    values: I,
    min: T,
    one: T,
    dither_amplitude: T,
    seed: u32,
) -> Vec<T>
where
    T: DitherFloat + Send + Sync,
    I: IntoIterator<Item = T>,
    I::IntoIter: Send,
{
    let method = Hash::new(seed);
    dither_iter_with(values, min, one, dither_amplitude, &method)
}

/// Dither values from an iterator using specific method.
#[cfg(not(feature = "rayon"))]
pub fn dither_iter_with<T, I, M: LinearRng>(
    values: I,
    min: T,
    one: T,
    dither_amplitude: T,
    method: &M,
) -> Vec<T>
where
    T: DitherFloat,
    I: IntoIterator<Item = T>,
{
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            dither_with(value, min, one, dither_amplitude, index as u32, method)
        })
        .collect()
}

#[cfg(feature = "rayon")]
pub fn dither_iter_with<T, I, M: LinearRng>(
    values: I,
    min: T,
    one: T,
    dither_amplitude: T,
    method: &M,
) -> Vec<T>
where
    T: DitherFloat + Send + Sync,
    I: IntoIterator<Item = T>,
    I::IntoIter: Send,
{
    use rayon::prelude::*;

    let values_vec: Vec<_> = values.into_iter().collect();
    values_vec
        .into_par_iter()
        .enumerate()
        .map(|(index, value)| {
            dither_with(value, min, one, dither_amplitude, index as u32, method)
        })
        .collect()
}

/// Simple dither for iterators using default hash method.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_iter<T, I>(values: I, one: T, seed: u32) -> Vec<T>
where
    T: DitherFloat + Number + CastableFrom<f32>,
    I: IntoIterator<Item = T>,
{
    let method = Hash::new(seed);
    simple_dither_iter_with(values, one, &method)
}

#[cfg(feature = "rayon")]
pub fn simple_dither_iter<T, I>(values: I, one: T, seed: u32) -> Vec<T>
where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
    I: IntoIterator<Item = T>,
    I::IntoIter: Send,
{
    let method = Hash::new(seed);
    simple_dither_iter_with(values, one, &method)
}

/// Simple dither for iterators using specific method.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_iter_with<T, I, M: LinearRng>(
    values: I,
    one: T,
    method: &M,
) -> Vec<T>
where
    T: DitherFloat + Number + CastableFrom<f32>,
    I: IntoIterator<Item = T>,
{
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            simple_dither_with(value, one, index as u32, method)
        })
        .collect()
}

#[cfg(feature = "rayon")]
pub fn simple_dither_iter_with<T, I, M: LinearRng>(
    values: I,
    one: T,
    method: &M,
) -> Vec<T>
where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
    I: IntoIterator<Item = T>,
    I::IntoIter: Send,
{
    use rayon::prelude::*;

    let values_vec: Vec<_> = values.into_iter().collect();
    values_vec
        .into_par_iter()
        .enumerate()
        .map(|(index, value)| {
            simple_dither_with(value, one, index as u32, method)
        })
        .collect()
}

/// Dither float values in a slice when converting to lower precision using
/// default hash method.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "nightly_f16")]
/// # {
/// use dithereens::dither_float_slice;
///
/// let values: Vec<f32> = vec![1.1, 1.2, 1.3, 1.4];
/// let result: Vec<f16> = dither_float_slice(&values, 42);
/// # }
/// ```
#[cfg(not(feature = "rayon"))]
pub fn dither_float_slice<Src, Dest>(values: &[Src], seed: u32) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64> + Copy,
{
    let method = Hash::new(seed);
    dither_float_slice_with(values, &method)
}

#[cfg(feature = "rayon")]
pub fn dither_float_slice<Src, Dest>(values: &[Src], seed: u32) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest>
        + DitherFloat
        + CastableFrom<f64>
        + Copy
        + Send
        + Sync,
    Dest: Send,
{
    let method = Hash::new(seed);
    dither_float_slice_with(values, &method)
}

/// Dither float values in a slice when converting to lower precision using a
/// specific linear RNG method.
#[cfg(not(feature = "rayon"))]
pub fn dither_float_slice_with<Src, Dest, M: LinearRng>(
    values: &[Src],
    method: &M,
) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64> + Copy,
{
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| dither_float_with(value, index as u32, method))
        .collect()
}

#[cfg(feature = "rayon")]
pub fn dither_float_slice_with<Src, Dest, M: LinearRng>(
    values: &[Src],
    method: &M,
) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest>
        + DitherFloat
        + CastableFrom<f64>
        + Copy
        + Send
        + Sync,
    Dest: Send,
{
    use rayon::prelude::*;

    values
        .par_iter()
        .enumerate()
        .map(|(index, &value)| dither_float_with(value, index as u32, method))
        .collect()
}

/// Dither float values in a 2D image slice when converting to lower precision
/// using a spatial RNG method.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "nightly_f16")]
/// # {
/// use dithereens::{InterleavedGradientNoise, dither_float_slice_2d};
///
/// let width = 256;
/// let pixels: Vec<f32> = vec![1.5; width * width];
/// let method = InterleavedGradientNoise::new(42);
/// let result: Vec<f16> = dither_float_slice_2d(&pixels, width, &method);
/// # }
/// ```
#[cfg(not(feature = "rayon"))]
pub fn dither_float_slice_2d<Src, Dest, M: SpatialRng>(
    values: &[Src],
    width: usize,
    method: &M,
) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64> + Copy,
{
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let x = (index % width) as u32;
            let y = (index / width) as u32;
            dither_float_2d(value, x, y, method)
        })
        .collect()
}

#[cfg(feature = "rayon")]
pub fn dither_float_slice_2d<Src, Dest, M: SpatialRng>(
    values: &[Src],
    width: usize,
    method: &M,
) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest>
        + DitherFloat
        + CastableFrom<f64>
        + Copy
        + Send
        + Sync,
    Dest: Send,
{
    use rayon::prelude::*;

    values
        .par_iter()
        .enumerate()
        .map(|(index, &value)| {
            let x = (index % width) as u32;
            let y = (index / width) as u32;
            dither_float_2d(value, x, y, method)
        })
        .collect()
}

/// Dither float values from an iterator when converting to lower precision
/// using default hash method.
#[cfg(not(feature = "rayon"))]
pub fn dither_float_iter<Src, Dest, I>(values: I, seed: u32) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64>,
    I: IntoIterator<Item = Src>,
{
    let method = Hash::new(seed);
    dither_float_iter_with(values, &method)
}

#[cfg(feature = "rayon")]
pub fn dither_float_iter<Src, Dest, I>(values: I, seed: u32) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest>
        + DitherFloat
        + CastableFrom<f64>
        + Send
        + Sync,
    Dest: Send,
    I: IntoIterator<Item = Src>,
    I::IntoIter: Send,
{
    let method = Hash::new(seed);
    dither_float_iter_with(values, &method)
}

/// Dither float values from an iterator when converting to lower precision
/// using a specific linear RNG method.
#[cfg(not(feature = "rayon"))]
pub fn dither_float_iter_with<Src, Dest, I, M: LinearRng>(
    values: I,
    method: &M,
) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest> + DitherFloat + CastableFrom<f64>,
    I: IntoIterator<Item = Src>,
{
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| dither_float_with(value, index as u32, method))
        .collect()
}

#[cfg(feature = "rayon")]
pub fn dither_float_iter_with<Src, Dest, I, M: LinearRng>(
    values: I,
    method: &M,
) -> Vec<Dest>
where
    Src: DitherFloatConversion<Dest>
        + DitherFloat
        + CastableFrom<f64>
        + Send
        + Sync,
    Dest: Send,
    I: IntoIterator<Item = Src>,
    I::IntoIter: Send,
{
    use rayon::prelude::*;

    let values_vec: Vec<_> = values.into_iter().collect();
    values_vec
        .into_par_iter()
        .enumerate()
        .map(|(index, value)| dither_float_with(value, index as u32, method))
        .collect()
}

/// Iterator adapter trait for dithering operations.
///
/// This trait provides methods to apply dithering directly to iterators,
/// allowing for chaining operations. The trait automatically handles index
/// tracking for deterministic dithering.
///
/// # Examples
///
/// ```rust
/// # use dithereens::DitherIteratorExt;
/// let values = vec![0.2f32, 0.5, 0.8];
///
/// let result: Vec<f32> = values
///     .iter()
///     .copied()
///     .map(|x| x * 0.5)
///     .simple_dither(255.0, 42);
/// ```
pub trait DitherIteratorExt<T>: Iterator<Item = T> + Sized
where
    T: DitherFloat,
{
    /// Apply dithering to all values in the iterator with full control.
    #[cfg(not(feature = "rayon"))]
    fn dither(self, min: T, one: T, dither_amplitude: T, seed: u32) -> Vec<T> {
        self.enumerate()
            .map(|(index, value)| {
                dither(value, min, one, dither_amplitude, index as u32, seed)
            })
            .collect()
    }

    #[cfg(feature = "rayon")]
    fn dither(self, min: T, one: T, dither_amplitude: T, seed: u32) -> Vec<T>
    where
        T: Send + Sync,
        Self: Send,
        Self::Item: Send,
    {
        dither_iter(self, min, one, dither_amplitude, seed)
    }

    /// Apply dithering with a specific method.
    #[cfg(not(feature = "rayon"))]
    fn dither_with<M: LinearRng>(
        self,
        min: T,
        one: T,
        dither_amplitude: T,
        method: &M,
    ) -> Vec<T> {
        self.enumerate()
            .map(|(index, value)| {
                dither_with(
                    value,
                    min,
                    one,
                    dither_amplitude,
                    index as u32,
                    method,
                )
            })
            .collect()
    }

    #[cfg(feature = "rayon")]
    fn dither_with<M: LinearRng>(
        self,
        min: T,
        one: T,
        dither_amplitude: T,
        method: &M,
    ) -> Vec<T>
    where
        T: Send + Sync,
        Self: Send,
        Self::Item: Send,
    {
        dither_iter_with(self, min, one, dither_amplitude, method)
    }

    /// Apply simple dithering to all values in the iterator.
    #[cfg(not(feature = "rayon"))]
    fn simple_dither(self, one: T, seed: u32) -> Vec<T>
    where
        T: Number + CastableFrom<f32>,
    {
        self.enumerate()
            .map(|(index, value)| simple_dither(value, one, index as u32, seed))
            .collect()
    }

    #[cfg(feature = "rayon")]
    fn simple_dither(self, one: T, seed: u32) -> Vec<T>
    where
        T: Number + CastableFrom<f32> + Send + Sync,
        Self: Send,
        Self::Item: Send,
    {
        simple_dither_iter(self, one, seed)
    }

    /// Apply simple dithering with a specific method.
    #[cfg(not(feature = "rayon"))]
    fn simple_dither_with<M: LinearRng>(self, one: T, method: &M) -> Vec<T>
    where
        T: Number + CastableFrom<f32>,
    {
        self.enumerate()
            .map(|(index, value)| {
                simple_dither_with(value, one, index as u32, method)
            })
            .collect()
    }

    #[cfg(feature = "rayon")]
    fn simple_dither_with<M: LinearRng>(self, one: T, method: &M) -> Vec<T>
    where
        T: Number + CastableFrom<f32> + Send + Sync,
        Self: Send,
        Self::Item: Send,
    {
        simple_dither_iter_with(self, one, method)
    }
}

/// Automatic implementation of DitherIteratorExt for all iterators yielding
/// DitherFloat types.
impl<I, T> DitherIteratorExt<T> for I
where
    I: Iterator<Item = T>,
    T: DitherFloat,
{
}

#[cfg(feature = "rayon")]
/// Parallel iterator adapter trait for dithering operations.
///
/// This trait provides methods to apply dithering directly to parallel
/// iterators, allowing for chaining operations with automatic parallelization.
///
/// # Examples
///
/// ```rust
/// # use dithereens::DitherParallelIteratorExt;
/// # use rayon::prelude::*;
/// let values = vec![0.2f32, 0.5, 0.8];
///
/// let result: Vec<f32> = values
///     .par_iter()
///     .copied()
///     .map(|x| x * 0.5)
///     .simple_dither(255.0, 42);
/// ```
pub trait DitherParallelIteratorExt<T>:
    rayon::iter::IndexedParallelIterator<Item = T> + Sized
where
    T: DitherFloat,
{
    /// Apply dithering to all values in the parallel iterator.
    fn dither(self, min: T, one: T, dither_amplitude: T, seed: u32) -> Vec<T>
    where
        T: Send + Sync,
    {
        use rayon::prelude::*;

        self.enumerate()
            .map(|(index, value)| {
                dither(value, min, one, dither_amplitude, index as u32, seed)
            })
            .collect()
    }

    /// Apply dithering with a specific method.
    fn dither_with<M>(
        self,
        min: T,
        one: T,
        dither_amplitude: T,
        method: &M,
    ) -> Vec<T>
    where
        T: Send + Sync,
        M: LinearRng + Sync,
    {
        use rayon::prelude::*;

        self.enumerate()
            .map(|(index, value)| {
                dither_with(
                    value,
                    min,
                    one,
                    dither_amplitude,
                    index as u32,
                    method,
                )
            })
            .collect()
    }

    /// Apply simple dithering to all values in the parallel iterator.
    fn simple_dither(self, one: T, seed: u32) -> Vec<T>
    where
        T: Number + CastableFrom<f32> + Send + Sync,
    {
        use rayon::prelude::*;

        self.enumerate()
            .map(|(index, value)| simple_dither(value, one, index as u32, seed))
            .collect()
    }

    /// Apply simple dithering with a specific method.
    fn simple_dither_with<M>(self, one: T, method: &M) -> Vec<T>
    where
        T: Number + CastableFrom<f32> + Send + Sync,
        M: LinearRng + Sync,
    {
        use rayon::prelude::*;

        self.enumerate()
            .map(|(index, value)| {
                simple_dither_with(value, one, index as u32, method)
            })
            .collect()
    }
}

#[cfg(feature = "rayon")]
/// Automatic implementation for all indexed parallel iterators yielding
/// DitherFloat types.
impl<I, T> DitherParallelIteratorExt<T> for I
where
    I: rayon::iter::IndexedParallelIterator<Item = T>,
    T: DitherFloat,
{
}

/// Position-based random number generation using spatial RNG methods.
///
/// This module provides deterministic, position-based random number generation
/// that is particularly useful for parallel processing where thread execution
/// order should not affect the results.
pub mod rng {
    use crate::SpatialRng;

    /// Generate a position-based random number in [0, 1] range.
    ///
    /// This function uses a spatial RNG to generate deterministic
    /// random numbers based on position. The same (x, y) coordinates with
    /// the same method will always produce the same result.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use dithereens::{SpatialHash, rng::spatial_rng};
    ///
    /// let method = SpatialHash::new(42);
    /// let random_value = spatial_rng(&method, 10, 20);
    /// assert!(random_value >= 0.0 && random_value <= 1.0);
    /// ```
    #[inline(always)]
    pub fn spatial_rng<T: SpatialRng>(method: &T, x: u32, y: u32) -> f32 {
        // Convert from [-1, 1] to [0, 1]
        (method.compute(x, y) + 1.0) * 0.5
    }

    /// Generate a position-based random number in [0, max] range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use dithereens::{InterleavedGradientNoise, rng::spatial_rng_range};
    ///
    /// let method = InterleavedGradientNoise::new(42);
    /// let random_value = spatial_rng_range(&method, 10, 20, 100.0);
    /// assert!(random_value >= 0.0 && random_value <= 100.0);
    /// ```
    #[inline(always)]
    pub fn spatial_rng_range<T: SpatialRng>(
        method: &T,
        x: u32,
        y: u32,
        max: f32,
    ) -> f32 {
        spatial_rng(method, x, y) * max
    }

    /// Generate a position-based random integer in [0, max) range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use dithereens::{SpatialHash, rng::spatial_rng_int};
    ///
    /// let method = SpatialHash::new(42);
    /// let random_int = spatial_rng_int(&method, 10, 20, 6); // dice roll
    /// assert!(random_int < 6);
    /// ```
    #[inline(always)]
    pub fn spatial_rng_int<T: SpatialRng>(
        method: &T,
        x: u32,
        y: u32,
        max: u32,
    ) -> u32 {
        (spatial_rng(method, x, y) * max as f32) as u32
    }
}
