//! 1D dithering methods for sequential data and images as flat arrays.
//!
//! This module provides linear (1D) dithering methods that process data
//! sequentially using only an index value. These are ideal for:
//! - Processing data streams or arrays in order.
//! - Dithering images as flat pixel arrays.
//! - When memory efficiency matters (no lookup tables).
//! - Consistent results with a given seed.

use crate::{DitherFloat, DitherFloatConversion, LinearRng};
#[cfg(not(feature = "nightly_f16"))]
use common_traits::{CastableFrom, Number};
#[cfg(feature = "nightly_f16")]
use common_traits_f16::{CastableFrom, Number};
use enum_dispatch::enum_dispatch;

/// Hash-based dithering (default method).
///
/// Fast general-purpose RNG with uniform distribution. Uses integer
/// hash mixing for speed. Good choice when you need consistent
/// performance across all index values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Hash {
    seed: u32,
}

impl Hash {
    /// Creates a new Hash-based dithering method with the given seed.
    ///
    /// The seed determines the noise pattern that will be generated.
    /// The same seed will always produce the same dithering pattern.
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }
}

impl LinearRng for Hash {
    #[inline(always)]
    fn compute(&self, index: u32) -> f32 {
        // Better mixing of index and seed.
        let mut hash = index;
        hash = hash.wrapping_mul(1664525).wrapping_add(self.seed);
        hash = hash.wrapping_mul(1664525).wrapping_add(1013904223);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;

        // Convert to [-1, 1] range.
        (hash as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// R2 low-discrepancy sequence for improved distribution.
///
/// Provides better spatial coverage than random sequences. Based on
/// the generalized golden ratio (1.32471...). Produces visually
/// pleasing patterns with minimal clustering or gaps.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct R2 {
    seed: f32,
}

impl Eq for R2 {}

impl core::hash::Hash for R2 {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        // Hash the bit representation of the float.
        self.seed.to_bits().hash(state);
    }
}

impl R2 {
    /// Creates a new R2 low-discrepancy sequence dithering method with the
    /// given seed.
    ///
    /// Uses the generalized golden ratio (R2 sequence) for better spatial
    /// distribution than random noise. The seed determines the starting
    /// point in the sequence.
    pub fn new(seed: u32) -> Self {
        Self {
            seed: seed as f32 * 0.618_034,
        }
    }
}

impl LinearRng for R2 {
    #[inline(always)]
    fn compute(&self, index: u32) -> f32 {
        // R2 sequence using generalized golden ratio.
        const ALPHA: f32 = 0.754_877_7; // 1/φ₂ where φ₂ = 1.32471795724474602596

        // Add seed as initial offset.
        let value = (self.seed + ALPHA * index as f32).fract();

        // Convert from [0, 1] to [-1, 1]
        value * 2.0 - 1.0
    }
}

/// Golden ratio sequence for 1D low-discrepancy sampling.
///
/// Classic low-discrepancy sequence using the golden ratio (1.618...).
/// Optimal for 1D coverage with the most uniform distribution possible.
/// Excellent for gradient-like data or smooth transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GoldenRatio {
    seed: f32,
}

impl Eq for GoldenRatio {}

impl core::hash::Hash for GoldenRatio {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        // Hash the bit representation of the float.
        self.seed.to_bits().hash(state);
    }
}

impl GoldenRatio {
    /// Creates a new Golden Ratio sequence dithering method with the given
    /// seed.
    ///
    /// Uses the golden ratio (φ = 1.618...) for optimal 1D low-discrepancy
    /// sampling. Provides the most uniform distribution for gradient-like
    /// data and smooth transitions.
    pub fn new(seed: u32) -> Self {
        Self {
            seed: seed as f32 * 0.381_966_02,
        }
    }
}

impl LinearRng for GoldenRatio {
    #[inline(always)]
    fn compute(&self, index: u32) -> f32 {
        const INV_GOLDEN: f32 = 0.618_034; // 1/φ where φ = 1.618033988749

        // Golden ratio sequence with seed offset.
        let value = (self.seed + INV_GOLDEN * index as f32).fract();

        // Convert from [0, 1] to [-1, 1]
        value * 2.0 - 1.0
    }
}

/// Enum for dynamic dispatch of 1D dithering methods.
///
/// This enum allows runtime selection of dithering methods. All variants
/// implement [`LinearRng`] through [`macro@enum_dispatch`], providing zero-cost
/// abstraction for dynamic method selection.
///
/// # Example
///
/// ```rust
/// use dithereens::{LinearDither, LinearRng};
///
/// let method = LinearDither::Hash(dithereens::Hash::new(42));
/// let noise = method.compute(100);
/// assert!(noise >= -1.0 && noise <= 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[enum_dispatch(LinearRng)]
pub enum LinearDither {
    /// Hash-based dithering method.
    Hash(Hash),
    /// R2 low-discrepancy sequence method.
    R2(R2),
    /// Golden ratio sequence method.
    GoldenRatio(GoldenRatio),
}
