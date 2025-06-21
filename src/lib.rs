//! Functions and traits for quantizing values with error-diffusion.
//!
//! Quantizing from `f32`/`f16` to `u16`/`u8` without dithering leads to.
//! banding. This crate provides dithering to reduce quantization artifacts.
//!
//! # Overview
//!
//! - **Single values**: [`dither()`], [`simple_dither()`].
//! - **Iterator processing**: [`dither_iter()`], [`simple_dither_iter()`].
//! - **In-place operations**: [`dither_slice()`], [`simple_dither_slice()`].
//! - **Iterator adapters**: [`DitherIteratorExt`] for method chaining.
//! - **Trait-based API**: [`Dither`], [`SimpleDither`] traits.
//! - **no_std support**: Works in embedded environments.
//! - **Generic types**: `f32`, `f64`, or any [`DitherFloat`] implementation.
//!
//! # Quick Start
//!
//! ```rust
//! # use dithereens::simple_dither;
//! let mut rng = rand::thread_rng();
//!
//! let value: f32 = 0.5;
//!
//! // Dither `value` to `127u8` or `128u8`, with a probability of 50%.
//! // Note that we still clamp the value since it could be outside.
//! // the target type's range.
//! let dithered_value: u8 =
//!     simple_dither(value, 255.0, &mut rng).clamp(0.0, 255.0) as u8;
//!
//! assert!(dithered_value == 127 || 128 == dithered_value);
//! ```
//!
//! # Iterator Adapters
//!
//! Use [`DitherIteratorExt`] for ergonomic method chaining:
//!
//! ```rust
//! # use dithereens::DitherIteratorExt;
//! # use rand::{SeedableRng, rngs::SmallRng};
//! let mut rng = SmallRng::seed_from_u64(42);
//! let pixel_values = vec![0.2f32, 0.5, 0.8, 0.1, 0.9];
//!
//! let result: Vec<f32> = pixel_values
//!     .iter()
//!     .copied()
//!     // +3/4 EV exposure.
//!     .map(|pixel| pixel * 2.0f32.powf(3.0 / 4.0))
//!     // Dither.
//!     .simple_dither(255.0, &mut rng);
//! ```
//!
//! # Performance Guide
//!
//! Based on benchmarks with 10,000 values:
//!
//! - **Single values**: [`dither()`], [`simple_dither()`].
//! - **In-place slice operations**: [`dither_slice()`],
//!   [`simple_dither_slice()`] (~5.6x faster than iterator methods)
//! - **Iterator chains**: [`dither_iter()`], [`simple_dither_iter()`], or
//!   [`DitherIteratorExt`] adapters (allocation overhead)
//!
//! # Parallel Processing
//!
//! Via `rayon` -- enabled by default.
//!
//! ```toml
//! [dependencies]
//! dithereens = { version = "0.1", features = ["rayon"] }
//! ```
//!
//! With `rayon` enabled, batch and slice functions use parallel processing..
//! RNG must implement `Rng + Send + Clone`..
//!
//! # `no_std` Support
//!
//! This crate supports `no_std` environments. The `libm` crate can be used to
//! pull in a possibly faster, native `round()` implementation. Otherwise a
//! manual implementation is used in `no_std` environments.
//!
//! ```toml
//! [dependencies]
//! # `no_std`
//! dithereens = { version = "0.1", default-features = false }
//! # Optional: uses `libm`'s `round()` function instead of a manual implementation for `no_std`.
//! dithereens = { version = "0.1", default-features = false, features = ["libm"] }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

/// Maximum number of threads to allocate on the stack for RNG distribution.
/// Set to 64 to handle most modern CPU core counts without heap allocation.
#[cfg(feature = "rayon")]
const MAX_THREADS: usize = 64;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use common_traits::{CastableFrom, Number};
#[cfg(feature = "rayon")]
use core::cell::UnsafeCell;
use core::{
    cmp::PartialOrd,
    ops::{Add, Mul, Neg, Sub},
};
use rand::{Rng, distr::uniform::SampleUniform};
#[cfg(feature = "rayon")]
use smallvec::SmallVec;

#[cfg(feature = "rayon")]
/// Thread-safe wrapper around UnsafeCell for RNG distribution
/// SAFETY: This is safe because we guarantee through modulo arithmetic
/// that no two threads will ever access the same RNG simultaneously.
struct UnsafeCellWrapper<R>(UnsafeCell<R>);

#[cfg(feature = "rayon")]
impl<R> UnsafeCellWrapper<R> {
    fn new(rng: R) -> Self {
        Self(UnsafeCell::new(rng))
    }

    /// SAFETY: Caller must guarantee exclusive access to this RNG.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut R {
        unsafe { &mut *self.0.get() }
    }
}

#[cfg(feature = "rayon")]
// SAFETY: We guarantee through modulo arithmetic that no two threads
// will ever access the same UnsafeCellWrapper simultaneously.
unsafe impl<R> Sync for UnsafeCellWrapper<R> where R: Send {}

#[cfg(feature = "rayon")]
unsafe impl<R> Send for UnsafeCellWrapper<R> where R: Send {}

#[cfg(feature = "rayon")]
/// Macro to generate rayon slice operations with shared RNG distribution
macro_rules! rayon_slice_op {
    ($values:expr, $rng_manager:expr, $operation:expr) => {{
        use rayon::prelude::*;

        // Pre-clone RNGs for each thread
        let num_threads = rayon::current_num_threads();
        let rngs: SmallVec<[UnsafeCellWrapper<_>; MAX_THREADS]> = (0
            ..num_threads)
            .map(|_| UnsafeCellWrapper::new($rng_manager.clone()))
            .collect();

        $values.par_iter_mut().enumerate().for_each(|(idx, value)| {
            let thread_id = idx % num_threads;
            // SAFETY: Each thread accesses a different RNG based on thread_id
            let rng = unsafe { rngs[thread_id].get_mut() };
            $operation(value, rng);
        });
    }};
}

#[cfg(feature = "rayon")]
/// Macro to generate rayon iterator operations with shared RNG distribution
macro_rules! rayon_iter_op {
    ($values:expr, $rng_manager:expr, $operation:expr) => {{
        use rayon::prelude::*;

        let values_vec = $values.into_iter().collect::<Vec<_>>();

        // Pre-clone RNGs for each thread
        let num_threads = rayon::current_num_threads();
        let rngs: SmallVec<[UnsafeCellWrapper<_>; MAX_THREADS]> = (0
            ..num_threads)
            .map(|_| UnsafeCellWrapper::new($rng_manager.clone()))
            .collect();

        values_vec
            .into_par_iter()
            .enumerate()
            .map(|(idx, value)| {
                let thread_id = idx % num_threads;
                // SAFETY: Each thread accesses a different RNG based on
                // thread_id
                let rng = unsafe { rngs[thread_id].get_mut() };
                $operation(value, rng)
            })
            .collect()
    }};
}

/// Minimal trait requirements for dithering
pub trait DitherFloat:
    Copy
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Number
    + SampleUniform
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

/// Trait for types that can be dithered.
///
/// Method-style API for the [`dither()`] function.
///
/// # Examples
///
/// ```rust
/// # use dithereens::Dither;
/// # use rand::thread_rng;
/// let mut rng = rand::thread_rng();
/// let value = 0.5_f32;
/// let dithered = value.dither(0.0, 255.0, 0.5, &mut rng);
/// ```
pub trait Dither<T>
where
    T: DitherFloat,
{
    /// Dither this value.
    ///
    /// See [`dither()`] for parameter details and examples.
    fn dither<R: Rng>(self, min: T, one: T, dither: T, rng: &mut R) -> T;
}

/// Trait for types that can be dithered using simplified parameters.
///
/// Method-style API for the [`simple_dither()`] function with sensible
/// defaults.
///
/// # Examples
///
/// ```rust
/// # use dithereens::SimpleDither;
/// # use rand::thread_rng;
/// let mut rng = rand::thread_rng();
/// let value = 0.5_f32;
/// let dithered = value.simple_dither(255.0, &mut rng);
/// ```
pub trait SimpleDither<T>
where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    /// Dither this value using simplified parameters.
    ///
    /// See [`simple_dither()`] for parameter details and examples.
    fn simple_dither<R: Rng>(self, one: T, rng: &mut R) -> T;
}

/// See the [`dither()`] function for more details.
impl<T> Dither<T> for T
where
    T: DitherFloat,
{
    #[inline]
    fn dither<R: Rng>(
        self,
        min: T,
        one: T,
        dither_amplitude: T,
        rng: &mut R,
    ) -> T {
        dither(self, min, one, dither_amplitude, rng)
    }
}

/// See the [`simple_dither()`] function for more details.
impl<T> SimpleDither<T> for T
where
    T: DitherFloat + Number + CastableFrom<f32>,
{
    #[inline]
    fn simple_dither<R: Rng>(self: T, one: T, rng: &mut R) -> T {
        simple_dither(self, one, rng)
    }
}

/// Dither a value using random noise.
///
/// Core dithering function with control over all parameters.
/// Result is *not* automatically clamped.
///
/// ## Algorithm
///
/// The function applies the formula: *⌊min + value × (one - min) + dither⌉*
///
/// Where `dither` is a random value in the range `[-dither_amplitude,
/// dither_amplitude]`.
///
/// ## Parameters
///
/// * `value` -- The input value to dither (typically in range 0.0..1.0)
/// * `min` -- The minimum output value (maps to input 0.0)
/// * `one` -- The maximum output value (maps to input 1.0)
/// * `dither_amplitude` -- The strength of dithering. Typical values:
///   - `0.5` for standard dithering
///   - `0.0` for no dithering (deterministic quantization)
///   - `1.0` for strong dithering (may cause visible noise)
/// * `rng` -- A random number generator
///
/// ## Performance
///
/// For multiple values, [`dither_slice()`] is fastest (in-place),
/// [`dither_iter()`] provides iterator flexibility with allocation overhead.
///
/// ## Examples
///
/// Basic usage:
/// ```
/// # use dithereens::dither;
/// let mut rng = rand::thread_rng();
///
/// let value: f32 = 0.5;
///
/// // Dither `value` to `127u8` or `128u8`, with a probability of 50%
/// let dithered_value: u8 =
///     dither(value, 0.0, 255.0, 0.5, &mut rng).clamp(0.0, 255.0) as u8;
///
/// assert!(dithered_value == 127 || 128 == dithered_value);
/// ```
///
/// Custom range mapping:
/// ```
/// # use dithereens::dither;
/// # use rand::thread_rng;
/// let mut rng = rand::thread_rng();
///
/// // Map 0.0..1.0 to -100.0..100.0 range
/// let result = dither(0.75, -100.0, 100.0, 0.5, &mut rng);
/// assert!(result >= 49.5 && result <= 50.5); // Around 50.0 ± dither
/// ```
#[inline]
pub fn dither<T, R>(
    value: T,
    min: T,
    one: T,
    dither_amplitude: T,
    rng: &mut R,
) -> T
where
    T: DitherFloat,
    R: Rng,
{
    let dither = if dither_amplitude == T::ZERO {
        T::ZERO
    } else {
        rng.random_range(-dither_amplitude..dither_amplitude)
    };

    (min + value * (one - min) + dither).round()
}

/// Dither a value using simplified parameters.
///
/// Convenience function with sensible defaults.
/// Maps `[0.0, 1.0]` to `[0.0, one]` with automatic clamping and dither
/// amplitude `0.5`.
///
/// ## Parameters
///
/// * `value` -- The input value to dither (any range, but typically 0.0..1.0)
/// * `one` -- The maximum output value (typically 255.0 for 8-bit, 65535.0 for
///   16-bit)
/// * `rng` -- A random number generator
///
/// ## Differences from [`dither()`]
///
/// - Uses `min = 0.0` (always maps to zero)
/// - Uses `dither_amplitude = 0.5` (standard strength)
/// - **Automatically clamps** output to `[0.0, one]` range
/// - Input values outside `[0.0, 1.0]` are handled gracefully
///
/// ## Performance
///
/// For multiple values, [`simple_dither_slice()`] is fastest (in-place, ~5.6x
/// faster), [`simple_dither_iter()`] provides iterator flexibility with
/// allocation overhead.
///
/// ## Examples
///
/// Standard usage for 8-bit quantization:
/// ```
/// # use dithereens::simple_dither;
/// let mut rng = rand::thread_rng();
///
/// let value: f32 = 0.5;
///
/// // Dither to 8-bit range - no manual clamping needed
/// let dithered_value: u8 = simple_dither(value, 255.0, &mut rng) as u8;
///
/// assert!(dithered_value == 127 || 128 == dithered_value);
/// ```
///
/// Processing HDR values (outside 0..1 range):
/// ```
/// # use dithereens::simple_dither;
/// # use rand::thread_rng;
/// let mut rng = rand::thread_rng();
///
/// // Input value outside normal range - automatically clamped
/// let result = simple_dither(1.5, 255.0, &mut rng);
/// assert!(result >= 0.0 && result <= 255.0);
/// ```
pub fn simple_dither<T, R>(value: T, one: T, rng: &mut R) -> T
where
    T: DitherFloat + Number + CastableFrom<f32>,
    R: Rng,
{
    dither(value, T::ZERO, one, T::cast_from(0.5_f32), rng).clamp(T::ZERO, one)
}

/// Dither multiple values in-place.
///
/// Fastest approach (~5.6x faster than iterator methods) with zero allocation.
/// Uses parallel processing when `rayon` feature is enabled.
///
/// # Examples
///
/// ```rust
/// # use dithereens::dither_slice;
/// # use rand::thread_rng;
/// let mut rng = rand::thread_rng();
/// let mut values = vec![0.2, 0.5, 0.8];
///
/// // Dither to 8-bit range.
/// dither_slice(&mut values, 0.0, 255.0, 0.5, &mut rng);
/// ```
#[cfg(not(feature = "rayon"))]
pub fn dither_slice<T, R>(
    values: &mut [T],
    min: T,
    one: T,
    dither_amplitude: T,
    rng: &mut R,
) where
    T: DitherFloat,
    R: Rng,
{
    for value in values.iter_mut() {
        *value = dither(*value, min, one, dither_amplitude, rng);
    }
}

#[cfg(feature = "rayon")]
pub fn dither_slice<T, R>(
    values: &mut [T],
    min: T,
    one: T,
    dither_amplitude: T,
    rng_manager: &R,
) where
    T: DitherFloat + Send + Sync,
    R: Rng + Send + Clone,
{
    rayon_slice_op!(values, rng_manager, |value: &mut T, rng: &mut R| {
        *value = dither(*value, min, one, dither_amplitude, rng);
    });
}

/// Dither multiple values in-place using simple dithering.
///
/// Zero allocation, most efficient for multiple values.
/// Uses parallel processing when `rayon` feature is enabled.
///
/// # Examples
///
/// ```rust
/// # use dithereens::simple_dither_slice;
/// # use rand::thread_rng;
/// let mut rng = rand::thread_rng();
/// let mut values = vec![0.2, 0.5, 0.8];
///
/// // Dither to 8-bit range.
/// simple_dither_slice(&mut values, 255.0, &mut rng);
/// ```
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_slice<T, R>(values: &mut [T], one: T, rng: &mut R)
where
    T: DitherFloat + Number + CastableFrom<f32>,
    R: Rng,
{
    for value in values.iter_mut() {
        *value = simple_dither(*value, one, rng);
    }
}

#[cfg(feature = "rayon")]
pub fn simple_dither_slice<T, R>(values: &mut [T], one: T, rng_manager: &R)
where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
    R: Rng + Send + Clone,
{
    rayon_slice_op!(values, rng_manager, |value: &mut T, rng: &mut R| {
        *value = simple_dither(*value, one, rng);
    });
}

/// Dither multiple values from any iterator.
///
/// Accepts `IntoIterator` (Vec, arrays, slices, iterator chains).
/// Has allocation overhead, ~5.6x slower than [`dither_slice()`].
/// Uses parallel processing when `rayon` feature is enabled.
///
/// # Examples
///
/// ```rust
/// # use dithereens::dither_iter;
/// # use rand::thread_rng;
/// let mut rng = rand::thread_rng();
///
/// // Works with Vec
/// let values = vec![0.2, 0.5, 0.8];
/// let dithered = dither_iter(values, 0.0, 255.0, 0.5, &mut rng);
///
/// // Works with arrays
/// let dithered = dither_iter([0.2, 0.5, 0.8], 0.0, 255.0, 0.5, &mut rng);
///
/// // Works with iterator chains
/// let dithered = dither_iter(
///     (0..10).map(|i| i as f32 / 10.0),
///     0.0,
///     255.0,
///     0.5,
///     &mut rng,
/// );
/// ```
#[cfg(not(feature = "rayon"))]
pub fn dither_iter<T, R, I>(
    values: I,
    min: T,
    one: T,
    dither_amplitude: T,
    rng: &mut R,
) -> Vec<T>
where
    T: DitherFloat,
    R: Rng,
    I: IntoIterator<Item = T>,
{
    values
        .into_iter()
        .map(|value| dither(value, min, one, dither_amplitude, rng))
        .collect()
}

#[cfg(feature = "rayon")]
pub fn dither_iter<T, R, I>(
    values: I,
    min: T,
    one: T,
    dither_amplitude: T,
    rng_manager: &R,
) -> Vec<T>
where
    T: DitherFloat + Send + Sync,
    R: Rng + Send + Clone,
    I: IntoIterator<Item = T>,
    I::IntoIter: Send,
    <I::IntoIter as Iterator>::Item: Send,
{
    rayon_iter_op!(values, rng_manager, |value: T, rng: &mut R| {
        dither(value, min, one, dither_amplitude, rng)
    })
}

/// Dither multiple values from any iterator using simple dithering.
///
/// Accepts `IntoIterator` (Vec, arrays, slices, iterator chains).
/// Has allocation overhead, ~5.6x slower than [`simple_dither_slice()`].
/// Uses parallel processing when `rayon` feature is enabled.
///
/// # Examples
///
/// ```rust
/// # use dithereens::simple_dither_iter;
/// # use rand::thread_rng;
/// let mut rng = rand::thread_rng();
///
/// // Works with Vec
/// let values = vec![0.2, 0.5, 0.8];
/// let dithered = simple_dither_iter(values, 255.0, &mut rng);
///
/// // Works with arrays
/// let dithered = simple_dither_iter([0.2, 0.5, 0.8], 255.0, &mut rng);
///
/// // Works with slice references
/// let slice = &[0.2, 0.5, 0.8][..];
/// let dithered = simple_dither_iter(slice.iter().copied(), 255.0, &mut rng);
/// ```
#[cfg(not(feature = "rayon"))]
pub fn simple_dither_iter<T, R, I>(values: I, one: T, rng: &mut R) -> Vec<T>
where
    T: DitherFloat + Number + CastableFrom<f32>,
    R: Rng,
    I: IntoIterator<Item = T>,
{
    values
        .into_iter()
        .map(|value| simple_dither(value, one, rng))
        .collect()
}

#[cfg(feature = "rayon")]
pub fn simple_dither_iter<T, R, I>(values: I, one: T, rng_manager: &R) -> Vec<T>
where
    T: DitherFloat + Number + CastableFrom<f32> + Send + Sync,
    R: Rng + Send + Clone,
    I: IntoIterator<Item = T>,
    I::IntoIter: Send,
    <I::IntoIter as Iterator>::Item: Send,
{
    rayon_iter_op!(values, rng_manager, |value: T, rng: &mut R| {
        simple_dither(value, one, rng)
    })
}

/// Iterator adapter trait for dithering operations.
///
/// This trait provides methods to apply dithering directly to iterators,
/// allowing for chaining operations like:
/// `values.iter().map(|x| x * 2.0).simple_dither(255.0, &mut rng)`
///
/// # Examples
///
/// ```rust
/// # use dithereens::DitherIteratorExt;
/// # use rand::{SeedableRng, rngs::SmallRng};
/// let mut rng = SmallRng::seed_from_u64(42);
/// let values = vec![0.2f32, 0.5, 0.8];
///
/// let result: Vec<f32> = values
///     .iter()
///     .copied()
///     .map(|x| x * 0.5)
///     .simple_dither(255.0, &mut rng);
/// ```
pub trait DitherIteratorExt<T>: Iterator<Item = T> + Sized
where
    T: DitherFloat,
{
    /// Apply dithering to all values in the iterator with full control.
    ///
    /// This is equivalent to calling [`dither_iter()`] but with a fluent API.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use dithereens::DitherIteratorExt;
    /// # use rand::{SeedableRng, rngs::SmallRng};
    /// let mut rng = SmallRng::seed_from_u64(42);
    /// let values = vec![0.2f32, 0.5, 0.8];
    ///
    /// let result: Vec<f32> =
    ///     values.iter().copied().dither(0.0, 255.0, 0.5, &mut rng);
    /// ```
    #[cfg(not(feature = "rayon"))]
    fn dither<R>(
        self,
        min: T,
        one: T,
        dither_amplitude: T,
        rng: &mut R,
    ) -> Vec<T>
    where
        R: Rng,
    {
        self.into_iter()
            .map(|value| dither(value, min, one, dither_amplitude, rng))
            .collect()
    }

    /// Apply dithering to all values in the iterator with full control (rayon
    /// version).
    #[cfg(feature = "rayon")]
    fn dither<R>(self, min: T, one: T, dither_amplitude: T, rng: &R) -> Vec<T>
    where
        T: Send + Sync,
        R: Rng + Send + Clone,
        Self: Send,
        Self::Item: Send,
    {
        dither_iter(self, min, one, dither_amplitude, rng)
    }

    /// Apply simple dithering to all values in the iterator.
    ///
    /// This is equivalent to calling [`simple_dither_iter()`] but with a fluent
    /// API.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use dithereens::DitherIteratorExt;
    /// # use rand::{SeedableRng, rngs::SmallRng};
    /// let mut rng = SmallRng::seed_from_u64(42);
    /// let values = vec![0.2f32, 0.5, 0.8];
    ///
    /// let result: Vec<f32> =
    ///     values.iter().copied().simple_dither(255.0, &mut rng);
    /// ```
    #[cfg(not(feature = "rayon"))]
    fn simple_dither<R>(self, one: T, rng: &mut R) -> Vec<T>
    where
        T: Number + CastableFrom<f32>,
        R: Rng,
    {
        self.into_iter()
            .map(|value| simple_dither(value, one, rng))
            .collect()
    }

    /// Apply simple dithering to all values in the iterator (rayon version).
    #[cfg(feature = "rayon")]
    fn simple_dither<R>(self, one: T, rng: &R) -> Vec<T>
    where
        T: Number + CastableFrom<f32> + Send + Sync,
        R: Rng + Send + Clone,
        Self: Send,
        Self::Item: Send,
    {
        simple_dither_iter(self, one, rng)
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
