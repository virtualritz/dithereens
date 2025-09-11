# CLAUDE.md -- dithereens -- Deterministic dithering functions and traits

## The Golden Rule

When unsure about implementation details, ALWAYS ask the developer.

## Project Context

This is a Rust crate providing deterministic hash-based dithering algorithms for quantizing floating-point values to integers with error diffusion. The crate supports both 1D and 2D dithering methods with optional parallelization via Rayon.

## Key Architecture Decisions

### API Breaking Change (v0.3.0)

The dithering method API was redesigned to pre-compute seed values for better performance:
- Methods now store seed-based values in their structs
- Seed is provided at construction time (`Method::new(seed)`) rather than at each `compute()` call
- This eliminates redundant calculations when processing multiple values

### Feature Flags

- `default = ["rayon", "std"]` - Standard library and parallel processing
- `std` - Enables std-only features  
- `libm` - Uses libm for `round()` in no_std environments
- `rayon` - Enables parallel processing
- `nightly_f16` - Native f16 type support (requires nightly)
- `blue_noise` - Real blue noise using 256x256x4 precomputed table (~5MB binary size)

### Dithering Methods

**1D Methods:**
- `Hash` - Default hash-based dithering
- `R2` - Low-discrepancy sequence
- `GoldenRatio` - Golden ratio sequence

**2D Methods:**
- `InterleavedGradientNoise` - IGN algorithm from Jorge Jimenez
- `SpatialHash` - Spatial hash for blue noise-like properties
- `BlueNoiseApprox` - Approximation using IGN + SpatialHash
- `BlueNoise` (requires `blue_noise` feature) - Real blue noise from precomputed table

## Code Style and Patterns

### Test Naming Convention
Test functions should NOT be prefixed with `test_`. The `#[test]` attribute already indicates it's a test.

### CRITICAL: Run tests before committing
Always run `cargo test --all-features` and ensure:
- All tests pass
- Code compiles without warnings
- Use `cargo fmt` to format code
- Run `cargo clippy --all-targets --all-features -- -W warnings` and fix issues

### Documentation
- All public APIs must have doc comments
- Include usage examples in doc comments where appropriate
- Use `cargo doc --open` to review documentation

## Build and Development Commands

```bash
# Build the project
cargo build

# Run tests
cargo test

# Test with all features including blue_noise
cargo test --all-features

# Run specific test
cargo test test_name

# Build with optimizations (only when explicitly needed)
cargo build --release

# Format code
cargo fmt

# Run clippy linter
cargo clippy --fix --allow-dirty

# Check code without building
cargo check

# Generate and view documentation
cargo doc --open

# Run examples
cargo run --example dither_2d
cargo run --example hash_dither
cargo run --example parallel_dither
cargo run --example iterator_adapters

# Run with blue_noise feature
cargo run --example dither_2d --features blue_noise
```

## Performance Requirements

- Use `#[inline(always)]` for hot path methods
- Prefer in-place operations (`_slice` functions) for best performance
- Iterator functions allocate - use slices when performance critical
- Rayon automatically parallelizes for large datasets

## API Design Patterns

- Functional API with method chaining support via iterator traits
- Deterministic - same input + seed always produces same output
- Generic over float types (f32, f64, f16 with feature)
- Separation of concerns: methods compute offsets, functions apply dithering

## Writing Instructions For User Interaction And Documentation

These instructions apply to any communication (e.g. feedback you print to the user) as well as any documentation you write.

- Be concise.
- AVOID weasel words.
- Use simple sentences. But feel free to use technical jargon.
- Do NOT overexplain basic concepts. Assume the user is technically proficient.
- AVOID flattering, corporate-ish or marketing language. Maintain a neutral viewpoint.
- AVOID vague and/or generic claims which may seem correct but are not substantiated by the context.

## Documentation

- All code comments MUST end with a period.
- All doc comments should also end with a period unless they're headlines. This includes list items.
- All list items in documentation MUST be complete sentences that end with a period.
- ENSURE an en-dash is expressed as two dashes like so: --. En-dashes are not used for connecting words, e.g. "compile-time".
- All references to types, keywords, symbols etc. MUST be enclosed in backticks: `struct` `Foo`.
- For each part of the docs, every first reference to a type, keyword, symbol etc. that is NOT the item itself that is being described MUST be linked to the relevant section in the docs like so: [`Foo`].
- NEVER use fully qualified paths in doc links. Use [`Foo`] instead of [`foo::bar::Foo`].

## What AI Must NEVER Do

1. **Never use --release unless explicitly requested** - Debug builds are sufficient for development
2. **Never modify test expectations** - Tests encode intended behavior
3. **Never remove doc comments** - Documentation is critical
4. **Never break public API without discussion** - This is a published crate
5. **Never add dependencies without approval** - Keep the crate lightweight

## Common Tasks

### Adding a New Dithering Method

1. Define struct with seed-based precomputed values
2. Implement constructor `new(seed: u32)`
3. Implement appropriate trait (`LinearRng` or `SpatialRng`)
4. Add tests comparing sequential vs parallel results
5. Update documentation and examples

### Debugging Test Failures

- Check if method instances are created with seeds
- Verify trait implementations match signatures
- For parallel tests, ensure determinism with same seed
- Use `cargo test -- --nocapture` to see print output

Remember: We optimize for correctness and determinism over raw speed. Users depend on getting the same results with the same inputs.