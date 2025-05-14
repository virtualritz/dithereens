//! Functions and traits for quantizing values with error-diffusion.
//!
//! This is mostly useful when e.g. quantizing from a `f32`- or
//! `f16`-per-channel color resolution to something like `u16`- or
//! `u8`-per-channel. In these cases quantization without error-diffusion
//! would lead to banding.
//!
//! The crate uses generics to allow interpolation of any type for which certain
//! traits are defined.
//!
//! ## Examples
//!
//! ```rust
//! # use num_traits::clamp;
//! # use dithereens::simple_dither;
//! let mut rng = rand::rng();
//!
//! let value: f32 = 0.5;
//!
//! // Dither `value` to `127u8` or `128u8``, with a probability of 50%.
//! //
//! // Note that we still clamp the value since it could be outside the target
//! // type's range.
//! let dithered_value: u8 =
//!     clamp(simple_dither(value, 255.0, &mut rng) as u8, 0, 255);
//!
//! assert!(dithered_value == 127 || 128 == dithered_value);
//! ```
use num_traits::{cast::FromPrimitive, clamp, float::Float, identities::Zero};
use rand::{Rng, distr::uniform::SampleUniform};

pub trait Dither<T>
where
    T: Float + SampleUniform,
{
    fn dither<R: Rng>(self, min: T, one: T, dither: T, rng: &mut R) -> T;
}

pub trait SimpleDither<T>
where
    T: Float + SampleUniform + Zero,
{
    fn simple_dither<R: Rng>(self, one: T, rng: &mut R) -> T;
}

/// See the [`dither()`] function for more details.
///
/// ## Examples
///
/// ```
/// # use dithereens::Dither;
/// let mut rng = rand::thread_rng();
///
/// let value = 0.5f32;
///
/// // Dither `value` to `127u8` or `128u8``, with a probability of 50%.
/// let dithered_value: u8 =
///     (value.dither(0.0, 255.0, 0.5, &mut rng) as u8).clamp(0, 255);
///
/// assert!(dithered_value == 127 || 128 == dithered_value);
/// ```
impl<T> Dither<T> for T
where
    T: Float + SampleUniform,
{
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
///
/// ## Examples
///
/// ```
/// # use dithereens::SimpleDither;
/// let mut rng = rand::thread_rng();
///
/// let value = 0.5f32;
///
/// // Dither `value` to `127u8` or `128u8``, with a probability of 50%.
/// let dithered_value: u8 =
///     (value.simple_dither(255.0, &mut rng) as u8).clamp(0, 255);
///
/// assert!(dithered_value == 127 || 128 == dithered_value);
/// ```
impl<T> SimpleDither<T> for T
where
    T: Float + FromPrimitive + SampleUniform + Zero,
{
    // Dither
    #[inline]
    fn simple_dither<R: Rng>(self: T, one: T, rng: &mut R) -> T {
        simple_dither(self, one, rng)
    }
}

/// Dither a value using random noise. With control over scaling, mapped
/// interval and dither strength.
///
/// *⌊min + value × (one - min) + dither⌉*
///
/// * `value` -- The value to dither
/// * `one` -- The value of 1.0 after dithering.
/// * `min..max` -- The range of `value` that is mapped to `0..1` before scaled
///   by `one` and dithered.
/// * `dither_amplitude` -- The amount/strength of dithering. A typical value is
///   `0.5`.
/// * `rng` -- A random number generator.
///
/// ## Examples
/// ```
/// # use num_traits::clamp;
/// # use dithereens::dither;
/// let mut rng = rand::thread_rng();
///
/// let value: f32 = 0.5;
///
/// // Dither `value` to `127u8` or `128u8``, with a probability of 50%.
/// //
/// // Note that we still clamp the value since it could be outside the target
/// // type's range.
/// let dithered_value: u8 =
///     clamp(dither(value, 0.0, 255.0, 0.5, &mut rng) as u8, 0, 255);
///
/// assert!(dithered_value == 127 || 128 == dithered_value);
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
    T: Float + SampleUniform,
    R: Rng,
{
    let dither = rng.random_range(-dither_amplitude..dither_amplitude);

    (min + value * (one - min) + dither).round()
}

/// Dither a value using random noise. With control over scaling.
///
/// See the [`dither()`] function for more details about the underlying math.
///
/// * `value` -- The value to dither
/// * `one` -- The value of 1.0 after dithering.
/// * `rng` -- A random number generator.
///
/// ## Examples
/// ```
/// # use num_traits::clamp;
/// # use dithereens::simple_dither;
/// let mut rng = rand::rng();
///
/// let value: f32 = 0.5;
///
/// // Dither `value` to `127u8` or `128u8``, with a probability of 50%.
/// //
/// // Note that we still clamp the value since it could be outside the target
/// // type's range.
/// let dithered_value: u8 =
///     clamp(simple_dither(value, 255.0, &mut rng) as u8, 0, 255);
///
/// assert!(dithered_value == 127 || 128 == dithered_value);
/// ```
pub fn simple_dither<T, R>(value: T, one: T, rng: &mut R) -> T
where
    T: Float + FromPrimitive + SampleUniform + Zero,
    R: Rng,
{
    clamp(
        dither(value, Zero::zero(), one, T::from_f32(0.5).unwrap(), rng),
        Zero::zero(),
        one,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dither() {
        use rand::{SeedableRng, rngs::SmallRng};

        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..100 {
            let value =
                (0.5_f32.dither(0.0, 255.0, 0.5, &mut rng) as u8).clamp(0, 255);
            assert!(value == 127 || 128 == value);
        }
    }
}
