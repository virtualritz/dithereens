use criterion::{Criterion, criterion_group, criterion_main};
use dithereens::{simple_dither, simple_dither_iter, simple_dither_slice};
use rand::{SeedableRng, rngs::SmallRng};
use std::hint::black_box;

fn bench_simple_comparison(c: &mut Criterion) {
    let values: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();

    c.bench_function("original_approach_1000", |b| {
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let results: Vec<f32> = values
                .iter()
                .map(|&v| simple_dither(black_box(v), 255.0, &mut rng))
                .collect();
            black_box(results)
        })
    });

    c.bench_function("batch_approach_1000", |b| {
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let results =
                simple_dither_iter(black_box(values.clone()), 255.0, &mut rng);
            black_box(results)
        })
    });

    c.bench_function("inplace_approach_1000", |b| {
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut values_copy = values.clone();
            simple_dither_slice(black_box(&mut values_copy), 255.0, &mut rng);
            black_box(values_copy)
        })
    });
}

criterion_group!(benches, bench_simple_comparison);
criterion_main!(benches);
