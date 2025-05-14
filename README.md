# `dithereens`

<!-- cargo-rdme start -->

Functions and traits for quantizing values with error-diffusion.

This is mostly useful when e.g. quantizing from a `f32`- or
`f16`-per-channel color resolution to something like `u16`- or
`u8`-per-channel. In these cases quantization without error-diffusion
would lead to banding.

The crate uses generics to allow interpolation of any type for which certain
traits are defined.

### Examples

```rust
let mut rng = rand::rng();

let value: f32 = 0.5;

// Dither `value` to `127u8` or `128u8``, with a probability of 50%.
//
// Note that we still clamp the value since it could be outside the target
// type's range.
let dithered_value: u8 =
    clamp(simple_dither(value, 255.0, &mut rng) as u8, 0, 255);

assert!(dithered_value == 127 || 128 == dithered_value);
```

<!-- cargo-rdme end -->

## License

Apache-2.0 OR BSD-3-Clause OR MIT OR Zlib at your discretion.
