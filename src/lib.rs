//! Functions and traits for quantizing values with deterministic hash-based
//! error-diffusion.
//!
//! Quantizing from `f64`/`f32`/`f16` to `u32`/`u16`/`u8` without dithering
//! creates banding. This crate provides deterministic hash-based dithering to
//! reduce quantization artifacts.
//!
//! # Overview
//!
//! - **Single values**: [`dither()`], [`simple_dither()`].
//! - **Iterator processing**: [`dither_iter()`], [`simple_dither_iter()`].
//! - **In-place operations**: [`dither_slice()`], [`simple_dither_slice()`].
//! - **`no_std` support**: Works in embedded environments.
//! - **Generic types**: `f32`, `f64`, `f16` (with `nightly_f16` feature), or
//!   any type implementing [`DitherFloat`].
//! - **Deterministic**: Same input with same seed always produces same output.
//!
//! # Quick Start
//!
//! ```rust
//! # use dithereens::simple_dither;
//!
//! let value: f32 = 0.5;
//!
//! // Dither `value` to `127u8` or `128u8` deterministically
//! // The same index and seed will always produce the same result
//! let dithered_value: u8 =
//!     simple_dither(value, 255.0, 0, 42).clamp(0.0, 255.0) as u8;
//!
//! assert!(dithered_value == 127 || 128 == dithered_value);
//! ```
//!
//! # Performance Guide
//!
//! Benchmarks with 10,000 values:
//!
//! - **Single values**: [`dither()`], [`simple_dither()`].
//! - **In-place slice operations**: [`dither_slice()`],
//!   [`simple_dither_slice()`] (fastest, zero allocation)
//! - **Iterator chains**: [`dither_iter()`], [`simple_dither_iter()`]
//!   (allocation overhead)
//!
//! # Parallel Processing
//!
//! Via `rayon` (enabled by default). With `rayon` enabled, `_iter` and
//! `_slice` functions use parallel processing automatically.
//!
//! # `no_std` Support
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
//! # Native `f16` Support
//!
//! Enable the `nightly_f16` feature to use native `f16` types (requires nightly
//! Rust):
//!
//! ```toml
//! [dependencies]
//! dithereens = { version = "0.3", features = ["nightly_f16"] }
//! ```
//!
//! # Blue Noise Support
//!
//! Enable the `blue_noise` feature to use true blue noise dithering with a
//! precomputed 256×256×4 table (adds ~5MB to binary size):
//!
//! ```toml
//! [dependencies]
//! dithereens = { version = "0.3", features = ["blue_noise"] }
//! ```
//!
//! This enables the [`BlueNoise`] struct which provides high-quality
//! blue noise dithering with stable seed-based variation.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly_f16", feature(f16))]

#[cfg(feature = "blue_noise")]
mod blue_noise;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use common_traits::{CastableFrom, Number};
use core::{
    cmp::PartialOrd,
    ops::{Add, Mul, Neg, Sub},
};

/// Trait for 1D dithering methods.
pub trait DitherMethod: Send + Sync {
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
        dither_with_method(value, min, one, dither_amplitude, index, self)
    }

    /// Simple dither for a single value.
    #[inline]
    fn simple_dither<T>(&self, value: T, one: T, index: u32) -> T
    where
        T: DitherFloat + Number + CastableFrom<f32>,
        Self: Sized,
    {
        simple_dither_with_method(value, one, index, self)
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
        dither_slice_with_method(values, min, one, dither_amplitude, self)
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
        dither_slice_with_method(values, min, one, dither_amplitude, self)
    }

    /// Simple dither for values in a slice.
    #[cfg(not(feature = "rayon"))]
    fn simple_dither_slice<T>(&self, values: &mut [T], one: T)
    where
        T: DitherFloat + Number + CastableFrom<f32>,
        Self: Sized,
    {
        simple_dither_slice_with_method(values, one, self)
    }

    /// Simple dither for values in a slice (parallel version).
    #[cfg(feature = "rayon")]
    fn simple_dither_slice<T>(&self, values: &mut [T], one: T)
    where
        T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
        Self: Sized,
    {
        simple_dither_slice_with_method(values, one, self)
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
        dither_iter_with_method(values, min, one, dither_amplitude, self)
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
        dither_iter_with_method(values, min, one, dither_amplitude, self)
    }

    /// Simple dither for values from an iterator.
    #[cfg(not(feature = "rayon"))]
    fn simple_dither_iter<T, I>(&self, values: I, one: T) -> Vec<T>
    where
        T: DitherFloat + Number + CastableFrom<f32>,
        I: IntoIterator<Item = T>,
        Self: Sized,
    {
        simple_dither_iter_with_method(values, one, self)
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
        simple_dither_iter_with_method(values, one, self)
    }
}

/// Trait for 2D dithering methods.
pub trait DitherMethod2D: Send + Sync {
    /// Compute dither offset for given 2D coordinates.
    /// Returns a value in range [-1, 1] to be scaled by amplitude.
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

    /// Dither a 2D image stored as a flat slice.
    #[cfg(not(feature = "rayon"))]
    fn dither_slice_2d<T>(
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
        dither_slice_2d(values, width, min, one, dither_amplitude, self)
    }

    /// Dither a 2D image stored as a flat slice (parallel version).
    #[cfg(feature = "rayon")]
    fn dither_slice_2d<T>(
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
        dither_slice_2d(values, width, min, one, dither_amplitude, self)
    }

    /// Simple dither for a 2D image stored as a flat slice.
    #[cfg(not(feature = "rayon"))]
    fn simple_dither_slice_2d<T>(&self, values: &mut [T], width: usize, one: T)
    where
        T: DitherFloat + Number + CastableFrom<f32>,
        Self: Sized,
    {
        simple_dither_slice_2d(values, width, one, self)
    }

    /// Simple dither for a 2D image stored as a flat slice (parallel version).
    #[cfg(feature = "rayon")]
    fn simple_dither_slice_2d<T>(&self, values: &mut [T], width: usize, one: T)
    where
        T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
        Self: Sized,
    {
        simple_dither_slice_2d(values, width, one, self)
    }
}

/// Hash-based dithering (default method).
pub struct Hash {
    seed: u32,
}

impl Hash {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }
}

impl DitherMethod for Hash {
    #[inline(always)]
    fn compute(&self, index: u32) -> f32 {
        // Better mixing of index and seed
        let mut hash = index;
        hash = hash.wrapping_mul(1664525).wrapping_add(self.seed);
        hash = hash.wrapping_mul(1664525).wrapping_add(1013904223);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;

        // Convert to [-1, 1] range
        (hash as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// R2 low-discrepancy sequence for improved distribution.
pub struct R2 {
    seed_offset: f32,
}

impl R2 {
    pub fn new(seed: u32) -> Self {
        Self {
            seed_offset: seed as f32 * 0.618_034,
        }
    }
}

impl DitherMethod for R2 {
    #[inline(always)]
    fn compute(&self, index: u32) -> f32 {
        // R2 sequence using generalized golden ratio
        const ALPHA: f32 = 0.754_877_7; // 1/φ₂ where φ₂ = 1.32471795724474602596

        // Add seed as initial offset
        let value = (self.seed_offset + ALPHA * index as f32).fract();

        // Convert from [0, 1] to [-1, 1]
        value * 2.0 - 1.0
    }
}

/// Golden ratio sequence for 1D low-discrepancy sampling.
pub struct GoldenRatio {
    seed_offset: f32,
}

impl GoldenRatio {
    pub fn new(seed: u32) -> Self {
        Self {
            seed_offset: seed as f32 * 0.381_966_02,
        }
    }
}

impl DitherMethod for GoldenRatio {
    #[inline(always)]
    fn compute(&self, index: u32) -> f32 {
        const INV_GOLDEN: f32 = 0.618_034; // 1/φ where φ = 1.618033988749

        // Golden ratio sequence with seed offset
        let value = (self.seed_offset + INV_GOLDEN * index as f32).fract();

        // Convert from [0, 1] to [-1, 1]
        value * 2.0 - 1.0
    }
}

/// Interleaved Gradient Noise for 2D dithering.
pub struct InterleavedGradientNoise {
    x_offset: u32,
    y_offset: u32,
}

impl InterleavedGradientNoise {
    pub fn new(seed: u32) -> Self {
        Self {
            x_offset: seed.wrapping_mul(5),
            y_offset: seed.wrapping_mul(7),
        }
    }
}

impl DitherMethod2D for InterleavedGradientNoise {
    #[inline(always)]
    fn compute(&self, x: u32, y: u32) -> f32 {
        // Add seed offset to coordinates
        let x_offset = x.wrapping_add(self.x_offset);
        let y_offset = y.wrapping_add(self.y_offset);

        // IGN algorithm from Jorge Jimenez
        let value = (52.982_918
            * ((0.06711056 * x_offset as f32 + 0.00583715 * y_offset as f32)
                .fract()))
        .fract();

        // Convert from [0, 1] to [-1, 1]
        value * 2.0 - 1.0
    }
}

/// Spatial hash for 2D blue noise-like properties.
pub struct SpatialHash {
    seed: u32,
}

impl SpatialHash {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }
}

impl DitherMethod2D for SpatialHash {
    #[inline(always)]
    fn compute(&self, x: u32, y: u32) -> f32 {
        // Combine x, y with good spatial decorrelation
        let mut hash = x;
        hash = hash.wrapping_mul(1664525).wrapping_add(y);
        hash = hash.wrapping_mul(1664525).wrapping_add(self.seed);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;

        // Convert to [-1, 1] range
        (hash as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Blue noise approximation using multiple octaves.
pub struct BlueNoiseApprox {
    ign: InterleavedGradientNoise,
    spatial: SpatialHash,
}

impl BlueNoiseApprox {
    pub fn new(seed: u32) -> Self {
        Self {
            ign: InterleavedGradientNoise::new(seed),
            spatial: SpatialHash::new(seed.wrapping_add(1337)),
        }
    }
}

impl DitherMethod2D for BlueNoiseApprox {
    #[inline(always)]
    fn compute(&self, x: u32, y: u32) -> f32 {
        // Use IGN as base with spatial hash for high-frequency detail
        let ign = self.ign.compute(x, y);
        let hash = self.spatial.compute(x >> 1, y >> 1);

        // Combine with emphasis on high frequencies
        (ign * 0.75 + hash * 0.25).clamp(-1.0, 1.0)
    }
}

/// True blue noise using precomputed table with stable seed-based offsetting.
#[cfg(feature = "blue_noise")]
pub struct BlueNoise {
    x_offset: u32,
    y_offset: u32,
    channel: usize,
}

#[cfg(feature = "blue_noise")]
impl BlueNoise {
    pub fn new(seed: u32) -> Self {
        Self {
            x_offset: seed.wrapping_mul(13),
            y_offset: seed.wrapping_mul(17),
            channel: ((seed >> 16) & 0x3) as usize,
        }
    }
}

#[cfg(feature = "blue_noise")]
impl DitherMethod2D for BlueNoise {
    #[inline(always)]
    fn compute(&self, x: u32, y: u32) -> f32 {
        // Apply precomputed seed-based offset to coordinates
        let x_offset = x.wrapping_add(self.x_offset);
        let y_offset = y.wrapping_add(self.y_offset);

        // Wrap coordinates to table size (256×256)
        let table_x = (x_offset & 0xFF) as usize;
        let table_y = (y_offset & 0xFF) as usize;

        // Access the precomputed blue noise table with preselected channel
        // Convert from [0, 1] to [-1, 1] range
        blue_noise::BLUE_NOISE_TABLE[table_y][table_x][self.channel] * 2.0 - 1.0
    }
}

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
    dither_with_method(value, min, one, dither_amplitude, index, &method)
}

/// Dither a value using a specific method.
#[inline]
pub fn dither_with_method<T, M: DitherMethod>(
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
    simple_dither_with_method(value, one, index, &method)
}

/// Simple dither with specific method.
#[inline]
pub fn simple_dither_with_method<T, M: DitherMethod>(
    value: T,
    one: T,
    index: u32,
    method: &M,
) -> T
where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    dither_with_method(
        value,
        T::ZERO,
        one,
        T::cast_from(0.5_f32),
        index,
        method,
    )
    .clamp(T::ZERO, one)
}

/// Dither a value using 2D coordinates.
#[inline]
pub fn dither_2d<T, M: DitherMethod2D>(
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
pub fn simple_dither_2d<T, M: DitherMethod2D>(
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
    dither_slice_with_method(values, min, one, dither_amplitude, &method)
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
    dither_slice_with_method(values, min, one, dither_amplitude, &method)
}

/// Dither values in a slice using specific method.
#[cfg(not(feature = "rayon"))]
pub fn dither_slice_with_method<T, M: DitherMethod>(
    values: &mut [T],
    min: T,
    one: T,
    dither_amplitude: T,
    method: &M,
) where
    T: DitherFloat,
{
    for (index, value) in values.iter_mut().enumerate() {
        *value = dither_with_method(
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
pub fn dither_slice_with_method<T, M: DitherMethod>(
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
            *value = dither_with_method(
                *value,
                min,
                one,
                dither_amplitude,
                index as u32,
                method,
            );
        });
}

/// Dither a 2D image stored as a flat slice.
#[cfg(not(feature = "rayon"))]
pub fn dither_slice_2d<T, M: DitherMethod2D>(
    values: &mut [T],
    width: usize,
    min: T,
    one: T,
    dither_amplitude: T,
    method: &M,
) where
    T: DitherFloat,
{
    for (index, value) in values.iter_mut().enumerate() {
        let x = (index % width) as u32;
        let y = (index / width) as u32;
        *value = dither_2d(*value, min, one, dither_amplitude, x, y, method);
    }
}

#[cfg(feature = "rayon")]
pub fn dither_slice_2d<T, M: DitherMethod2D>(
    values: &mut [T],
    width: usize,
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
            let x = (index % width) as u32;
            let y = (index / width) as u32;
            *value =
                dither_2d(*value, min, one, dither_amplitude, x, y, method);
        });
}

/// Simple dither for slices using default hash method.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_slice<T>(values: &mut [T], one: T, seed: u32)
where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    let method = Hash::new(seed);
    simple_dither_slice_with_method(values, one, &method)
}

#[cfg(feature = "rayon")]
pub fn simple_dither_slice<T>(values: &mut [T], one: T, seed: u32)
where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
{
    let method = Hash::new(seed);
    simple_dither_slice_with_method(values, one, &method)
}

/// Simple dither for slices using specific method.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_slice_with_method<T, M: DitherMethod>(
    values: &mut [T],
    one: T,
    method: &M,
) where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    for (index, value) in values.iter_mut().enumerate() {
        *value = simple_dither_with_method(*value, one, index as u32, method);
    }
}

#[cfg(feature = "rayon")]
pub fn simple_dither_slice_with_method<T, M: DitherMethod>(
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
            *value =
                simple_dither_with_method(*value, one, index as u32, method);
        });
}

/// Simple dither for 2D slices.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_slice_2d<T, M: DitherMethod2D>(
    values: &mut [T],
    width: usize,
    one: T,
    method: &M,
) where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    for (index, value) in values.iter_mut().enumerate() {
        let x = (index % width) as u32;
        let y = (index / width) as u32;
        *value = simple_dither_2d(*value, one, x, y, method);
    }
}

#[cfg(feature = "rayon")]
pub fn simple_dither_slice_2d<T, M: DitherMethod2D>(
    values: &mut [T],
    width: usize,
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
            let x = (index % width) as u32;
            let y = (index / width) as u32;
            *value = simple_dither_2d(*value, one, x, y, method);
        });
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
    dither_iter_with_method(values, min, one, dither_amplitude, &method)
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
    dither_iter_with_method(values, min, one, dither_amplitude, &method)
}

/// Dither values from an iterator using specific method.
#[cfg(not(feature = "rayon"))]
pub fn dither_iter_with_method<T, I, M: DitherMethod>(
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
            dither_with_method(
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
pub fn dither_iter_with_method<T, I, M: DitherMethod>(
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
            dither_with_method(
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

/// Simple dither for iterators using default hash method.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_iter<T, I>(values: I, one: T, seed: u32) -> Vec<T>
where
    T: DitherFloat + Number + CastableFrom<f32>,
    I: IntoIterator<Item = T>,
{
    let method = Hash::new(seed);
    simple_dither_iter_with_method(values, one, &method)
}

#[cfg(feature = "rayon")]
pub fn simple_dither_iter<T, I>(values: I, one: T, seed: u32) -> Vec<T>
where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
    I: IntoIterator<Item = T>,
    I::IntoIter: Send,
{
    let method = Hash::new(seed);
    simple_dither_iter_with_method(values, one, &method)
}

/// Simple dither for iterators using specific method.
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_iter_with_method<T, I, M: DitherMethod>(
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
            simple_dither_with_method(value, one, index as u32, method)
        })
        .collect()
}

#[cfg(feature = "rayon")]
pub fn simple_dither_iter_with_method<T, I, M: DitherMethod>(
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
            simple_dither_with_method(value, one, index as u32, method)
        })
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
    fn dither_with_method<M: DitherMethod>(
        self,
        min: T,
        one: T,
        dither_amplitude: T,
        method: &M,
    ) -> Vec<T> {
        self.enumerate()
            .map(|(index, value)| {
                dither_with_method(
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
    fn dither_with_method<M: DitherMethod>(
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
        dither_iter_with_method(self, min, one, dither_amplitude, method)
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
    fn simple_dither_with_method<M: DitherMethod>(
        self,
        one: T,
        method: &M,
    ) -> Vec<T>
    where
        T: Number + CastableFrom<f32>,
    {
        self.enumerate()
            .map(|(index, value)| {
                simple_dither_with_method(value, one, index as u32, method)
            })
            .collect()
    }

    #[cfg(feature = "rayon")]
    fn simple_dither_with_method<M: DitherMethod>(
        self,
        one: T,
        method: &M,
    ) -> Vec<T>
    where
        T: Number + CastableFrom<f32> + Send + Sync,
        Self: Send,
        Self::Item: Send,
    {
        simple_dither_iter_with_method(self, one, method)
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
    fn dither_with_method<M>(
        self,
        min: T,
        one: T,
        dither_amplitude: T,
        method: &M,
    ) -> Vec<T>
    where
        T: Send + Sync,
        M: DitherMethod + Sync,
    {
        use rayon::prelude::*;

        self.enumerate()
            .map(|(index, value)| {
                dither_with_method(
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
    fn simple_dither_with_method<M>(
        self,
        one: T,
        method: &M,
    ) -> Vec<T>
    where
        T: Number + CastableFrom<f32> + Send + Sync,
        M: DitherMethod + Sync,
    {
        use rayon::prelude::*;

        self.enumerate()
            .map(|(index, value)| {
                simple_dither_with_method(value, one, index as u32, method)
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
