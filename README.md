# `dithereens`

Functions and traits for quantizing values with error-diffusion.

This is mostly useful when e.g. quantizing from a `f32`- or `f16`-per-channel color resolution to something like `u16`- or `u8`-per-channel. In these cases quantization without error-diffusion would lead to banding.

The crate uses generics to allow interpolation of any type for which certain traits are defined.
