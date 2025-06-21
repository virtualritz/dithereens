use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dithereens::{simple_dither, simple_dither_iter, simple_dither_slice};
use rand::{SeedableRng, rngs::SmallRng};
use std::hint::black_box;

fn bench_performance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_comparison");

    for size in [100, 1000, 10000].iter() {
        let values: Vec<f32> =
            (0..*size).map(|i| i as f32 / *size as f32).collect();

        // Original approach: individual calls
        group.bench_with_input(
            BenchmarkId::new("individual_calls", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = SmallRng::seed_from_u64(42);
                    let results: Vec<f32> = values
                        .iter()
                        .map(|&v| simple_dither(black_box(v), 255.0, &mut rng))
                        .collect();
                    black_box(results)
                })
            },
        );

        // New batch function
        group.bench_with_input(
            BenchmarkId::new("batch_function", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = SmallRng::seed_from_u64(42);
                    let results = simple_dither_iter(
                        black_box(values.clone()),
                        255.0,
                        &mut rng,
                    );
                    black_box(results)
                })
            },
        );

        // New in-place function (most optimized)
        group.bench_with_input(
            BenchmarkId::new("inplace_function", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = SmallRng::seed_from_u64(42);
                    let mut values_copy = values.clone();
                    simple_dither_slice(
                        black_box(&mut values_copy),
                        255.0,
                        &mut rng,
                    );
                    black_box(values_copy)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_performance_comparison);
criterion_main!(benches);
