//! 2D dithering methods for images using spatial coordinates.
//!
//! This module provides spatial (2D) dithering methods that use pixel
//! coordinates to create spatially-aware dithering patterns. These methods are
//! ideal for:
//! - Image dithering with coordinate-based noise.
//! - Spatially decorrelated patterns.
//! - Blue noise-like characteristics.
//! - Visually pleasing results for images.

use crate::{DitherFloat, DitherFloatConversion, SpatialRng};
#[cfg(not(feature = "nightly_f16"))]
use common_traits::{CastableFrom, Number};
#[cfg(feature = "nightly_f16")]
use common_traits_f16::{CastableFrom, Number};
use enum_dispatch::enum_dispatch;

/// Interleaved Gradient Noise for 2D dithering.
///
/// Fast algorithm from Jorge Jimenez's presentation at SIGGRAPH 2014.
/// Widely used in real-time graphics for its speed and quality balance.
/// Creates smooth gradient-like patterns with good visual properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InterleavedGradientNoise {
    x_offset: u32,
    y_offset: u32,
}

impl InterleavedGradientNoise {
    /// Creates a new Interleaved Gradient Noise (IGN) dithering method with the
    /// given seed.
    ///
    /// Uses the fast IGN algorithm from Jorge Jimenez's SIGGRAPH 2014
    /// presentation. The seed determines the coordinate offsets for spatial
    /// variation.
    pub fn new(seed: u32) -> Self {
        Self {
            x_offset: seed.wrapping_mul(5),
            y_offset: seed.wrapping_mul(7),
        }
    }
}

impl SpatialRng for InterleavedGradientNoise {
    #[inline(always)]
    fn compute(&self, x: u32, y: u32) -> f32 {
        // Add seed offset to coordinates.
        let x_offset = x.wrapping_add(self.x_offset);
        let y_offset = y.wrapping_add(self.y_offset);

        // IGN algorithm from Jorge Jimenez.
        let value = (52.982_918
            * ((0.06711056 * x_offset as f32 + 0.00583715 * y_offset as f32)
                .fract()))
        .fract();

        // Convert from [0, 1] to [-1, 1]
        value * 2.0 - 1.0
    }
}

/// Spatial hash for 2D blue noise-like properties.
///
/// Uses coordinate hashing to create spatially decorrelated noise.
/// Provides blue noise-like characteristics without lookup tables.
/// Good balance between quality and memory efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialHash {
    seed: u32,
}

impl SpatialHash {
    /// Creates a new Spatial Hash dithering method with the given seed.
    ///
    /// Uses coordinate hashing to create spatially decorrelated noise with
    /// blue noise-like characteristics without lookup tables.
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }
}

impl SpatialRng for SpatialHash {
    #[inline(always)]
    fn compute(&self, x: u32, y: u32) -> f32 {
        // Combine x, y with good spatial decorrelation.
        let mut hash = x;
        hash = hash.wrapping_mul(1664525).wrapping_add(y);
        hash = hash.wrapping_mul(1664525).wrapping_add(self.seed);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;

        // Convert to [-1, 1] range.
        (hash as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Blue noise approximation using multiple octaves.
///
/// Hybrid approach that combines [`InterleavedGradientNoise`] with
/// [`SpatialHash`] to approximate true blue noise characteristics.
/// Better quality than either method alone, without the memory cost
/// of real blue noise tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlueNoiseApprox {
    ign: InterleavedGradientNoise,
    spatial: SpatialHash,
}

impl BlueNoiseApprox {
    /// Creates a new Blue Noise Approximation dithering method with the given
    /// seed.
    ///
    /// Combines IGN and SpatialHash to approximate blue noise characteristics
    /// without the memory cost of precomputed tables.
    pub fn new(seed: u32) -> Self {
        Self {
            ign: InterleavedGradientNoise::new(seed),
            spatial: SpatialHash::new(seed.wrapping_add(1337)),
        }
    }
}

impl SpatialRng for BlueNoiseApprox {
    #[inline(always)]
    fn compute(&self, x: u32, y: u32) -> f32 {
        // Use IGN as base with spatial hash for high-frequency detail.
        let ign = self.ign.compute(x, y);
        let hash = self.spatial.compute(x >> 1, y >> 1);

        // Combine with emphasis on high frequencies
        (ign * 0.75 + hash * 0.25).clamp(-1.0, 1.0)
    }
}

/// True blue noise using precomputed table with stable seed-based offsetting.
///
/// Highest quality dithering using true blue noise from precomputed
/// tables. Blue noise has optimal spectral characteristics -- high
/// frequency content with no low-frequency clustering. Results in
/// the most visually pleasing dithering patterns. Adds ~5MB to binary size.
#[cfg(feature = "blue-noise")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlueNoise {
    x_offset: u32,
    y_offset: u32,
    channel: usize,
}

#[cfg(feature = "blue-noise")]
impl BlueNoise {
    /// Creates a new true Blue Noise dithering method with the given seed.
    ///
    /// Uses precomputed 256×256×4 blue noise tables for highest quality
    /// dithering. The seed determines which channel and coordinate offsets
    /// to use.
    pub fn new(seed: u32) -> Self {
        Self {
            x_offset: seed.wrapping_mul(13),
            y_offset: seed.wrapping_mul(17),
            channel: ((seed >> 16) & 0x3) as usize,
        }
    }
}

#[cfg(feature = "blue-noise")]
impl SpatialRng for BlueNoise {
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
        crate::blue_noise::BLUE_NOISE_TABLE[table_y][table_x][self.channel]
            * 2.0
            - 1.0
    }
}

/// Enum for dynamic dispatch of 2D dithering methods.
///
/// This enum allows runtime selection of spatial dithering methods. All
/// variants implement [`SpatialRng`] through [`macro@enum_dispatch`], providing
/// zero-cost abstraction for dynamic method selection.
///
/// # Example
///
/// ```rust
/// use dithereens::{InterleavedGradientNoise, SpatialDither, SpatialRng};
///
/// let method = SpatialDither::InterleavedGradientNoise(
///     InterleavedGradientNoise::new(42),
/// );
/// let noise = method.compute(10, 20);
/// assert!(noise >= -1.0 && noise <= 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[enum_dispatch(SpatialRng)]
pub enum SpatialDither {
    /// Interleaved Gradient Noise method.
    InterleavedGradientNoise(InterleavedGradientNoise),
    /// Spatial hash method.
    SpatialHash(SpatialHash),
    /// Blue noise approximation method.
    BlueNoiseApprox(BlueNoiseApprox),
    /// True blue noise from precomputed tables (requires `blue-noise` feature).
    #[cfg(feature = "blue-noise")]
    BlueNoise(BlueNoise),
}
