use dithereens::{
    GoldenRatio, Hash, InterleavedGradientNoise, LinearDither, LinearRng, R2,
    SpatialDither, SpatialHash, SpatialRng,
};

#[test]
fn linear_dither_enum_hash() {
    let method = LinearDither::Hash(Hash::new(42));
    let noise = method.compute(100);
    assert!(noise >= -1.0 && noise <= 1.0);

    // Verify it produces the same result as the direct type
    let hash = Hash::new(42);
    assert_eq!(noise, hash.compute(100));
}

#[test]
fn linear_dither_enum_r2() {
    let method = LinearDither::R2(R2::new(42));
    let noise = method.compute(100);
    assert!(noise >= -1.0 && noise <= 1.0);

    // Verify it produces the same result as the direct type
    let r2 = R2::new(42);
    assert_eq!(noise, r2.compute(100));
}

#[test]
fn linear_dither_enum_golden_ratio() {
    let method = LinearDither::GoldenRatio(GoldenRatio::new(42));
    let noise = method.compute(100);
    assert!(noise >= -1.0 && noise <= 1.0);

    // Verify it produces the same result as the direct type
    let golden = GoldenRatio::new(42);
    assert_eq!(noise, golden.compute(100));
}

#[test]
fn spatial_dither_enum_ign() {
    let method = SpatialDither::InterleavedGradientNoise(
        InterleavedGradientNoise::new(42),
    );
    let noise = method.compute(10, 20);
    assert!(noise >= -1.0 && noise <= 1.0);

    // Verify it produces the same result as the direct type
    let ign = InterleavedGradientNoise::new(42);
    assert_eq!(noise, ign.compute(10, 20));
}

#[test]
fn spatial_dither_enum_spatial_hash() {
    let method = SpatialDither::SpatialHash(SpatialHash::new(42));
    let noise = method.compute(10, 20);
    assert!(noise >= -1.0 && noise <= 1.0);

    // Verify it produces the same result as the direct type
    let spatial = SpatialHash::new(42);
    assert_eq!(noise, spatial.compute(10, 20));
}

#[test]
fn linear_dither_dynamic_dispatch() {
    // Test dynamic dispatch with a vector of different methods
    let methods: Vec<LinearDither> = vec![
        LinearDither::Hash(Hash::new(1)),
        LinearDither::R2(R2::new(2)),
        LinearDither::GoldenRatio(GoldenRatio::new(3)),
    ];

    for method in &methods {
        let noise = method.compute(50);
        assert!(noise >= -1.0 && noise <= 1.0);
    }
}

#[test]
fn spatial_dither_dynamic_dispatch() {
    // Test dynamic dispatch with a vector of different methods
    let methods: Vec<SpatialDither> = vec![
        SpatialDither::InterleavedGradientNoise(InterleavedGradientNoise::new(
            1,
        )),
        SpatialDither::SpatialHash(SpatialHash::new(2)),
    ];

    for method in &methods {
        let noise = method.compute(15, 25);
        assert!(noise >= -1.0 && noise <= 1.0);
    }
}

#[test]
fn linear_dither_in_function() {
    // Test passing enum to a generic function
    fn process_with_dither<T: LinearRng>(method: &T, index: u32) -> f32 {
        method.compute(index)
    }

    let hash_enum = LinearDither::Hash(Hash::new(42));
    let r2_enum = LinearDither::R2(R2::new(42));

    let hash_result = process_with_dither(&hash_enum, 100);
    let r2_result = process_with_dither(&r2_enum, 100);

    assert!(hash_result >= -1.0 && hash_result <= 1.0);
    assert!(r2_result >= -1.0 && r2_result <= 1.0);
    assert_ne!(hash_result, r2_result); // Different methods should give different results
}

#[test]
fn spatial_dither_in_function() {
    // Test passing enum to a generic function
    fn process_with_dither<T: SpatialRng>(method: &T, x: u32, y: u32) -> f32 {
        method.compute(x, y)
    }

    let ign_enum = SpatialDither::InterleavedGradientNoise(
        InterleavedGradientNoise::new(42),
    );
    let spatial_enum = SpatialDither::SpatialHash(SpatialHash::new(42));

    let ign_result = process_with_dither(&ign_enum, 10, 20);
    let spatial_result = process_with_dither(&spatial_enum, 10, 20);

    assert!(ign_result >= -1.0 && ign_result <= 1.0);
    assert!(spatial_result >= -1.0 && spatial_result <= 1.0);
    assert_ne!(ign_result, spatial_result); // Different methods should give different results
}

#[test]
fn linear_dither_deterministic() {
    let method = LinearDither::Hash(Hash::new(42));

    // Same input should always produce same output
    let result1 = method.compute(123);
    let result2 = method.compute(123);
    assert_eq!(result1, result2);

    // Different inputs should produce different outputs (most of the time)
    let result3 = method.compute(124);
    assert_ne!(result1, result3);
}

#[test]
fn spatial_dither_deterministic() {
    let method = SpatialDither::SpatialHash(SpatialHash::new(42));

    // Same input should always produce same output
    let result1 = method.compute(10, 20);
    let result2 = method.compute(10, 20);
    assert_eq!(result1, result2);

    // Different inputs should produce different outputs
    let result3 = method.compute(10, 21);
    assert_ne!(result1, result3);
}
