# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2025-10-28

### Changed - BREAKING

- **Multi-channel API redesign**: 2D dithering functions now use const-generic parameters for channel count and noise correlation
  - `simple_dither_slice_2d::<CHANNELS, SEED_OFFSET, T, M>(...)`
  - `dither_slice_2d::<CHANNELS, SEED_OFFSET, T, M>(...)`
  - `CHANNELS`: Number of channels per pixel (1 = grayscale, 3 = RGB, 4 = RGBA)
  - `SEED_OFFSET`: Controls per-channel noise correlation
    - `0` = Correlated noise (same across all channels) -- fastest, ~CHANNELS× fewer noise computations
    - `>0` = Uncorrelated noise (different per channel) -- more randomness

- **Migration guide**:
  ```rust
  // Before (v0.5.0):
  simple_dither_slice_2d(&mut pixels, width, 255.0, &method);

  // After (v0.6.0):
  // For grayscale/single-channel:
  simple_dither_slice_2d::<1, 0, _, _>(&mut pixels, width, 255.0, &method);

  // For RGB with correlated noise:
  simple_dither_slice_2d::<3, 0, _, _>(&mut rgb_data, width, 255.0, &method);

  // For RGB with uncorrelated noise:
  simple_dither_slice_2d::<3, 1, _, _>(&mut rgb_data, width, 255.0, &method);
  ```

### Added

- Efficient multi-channel image dithering with noise reuse across channels
- Rayon parallelization now processes rows in parallel for better performance on multi-channel images
- New benchmark suite (`benches/multichannel.rs`) comparing correlated vs uncorrelated performance
- Comprehensive multi-channel examples in README and documentation

### Performance

- **RGB images with correlated noise**: Up to 3× fewer noise computations vs processing each channel separately
- **RGBA images with correlated noise**: Up to 4× fewer noise computations
- Better cache locality when processing multi-channel pixels together
- Rayon scaling improvements for large multi-channel images

## [0.5.0] - Previous release

(Previous changelog entries not available)
