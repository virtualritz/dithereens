use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dithereens::{Dither, SimpleDither, dither, simple_dither};
use rand::{SeedableRng, rngs::SmallRng};
use std::hint::black_box;

fn bench_dither_single(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    c.bench_function("dither_single_f32", |b| {
        b.iter(|| {
            let value = black_box(0.5f32);
            dither(value, 0.0, 255.0, 0.5, &mut rng)
        })
    });

    c.bench_function("simple_dither_single_f32", |b| {
        b.iter(|| {
            let value = black_box(0.5f32);
            simple_dither(value, 255.0, &mut rng)
        })
    });
}

fn bench_dither_trait_methods(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    c.bench_function("dither_trait_method", |b| {
        b.iter(|| {
            let value = black_box(0.5f32);
            value.dither(0.0, 255.0, 0.5, &mut rng)
        })
    });

    c.bench_function("simple_dither_trait_method", |b| {
        b.iter(|| {
            let value = black_box(0.5f32);
            value.simple_dither(255.0, &mut rng)
        })
    });
}

fn bench_dither_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("dither_iter");

    for size in [100, 1000, 10000].iter() {
        let values: Vec<f32> =
            (0..*size).map(|i| i as f32 / *size as f32).collect();

        group.bench_with_input(
            BenchmarkId::new("simple_dither", size),
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

        group.bench_with_input(
            BenchmarkId::new("dither", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = SmallRng::seed_from_u64(42);
                    let results: Vec<f32> = values
                        .iter()
                        .map(|&v| {
                            dither(black_box(v), 0.0, 255.0, 0.5, &mut rng)
                        })
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

fn bench_different_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("dither_types");
    let mut rng = SmallRng::seed_from_u64(42);

    group.bench_function("f32", |b| {
        b.iter(|| {
            let value = black_box(0.5f32);
            simple_dither(value, 255.0, &mut rng)
        })
    });

    group.bench_function("f64", |b| {
        b.iter(|| {
            let value = black_box(0.5f64);
            simple_dither(value, 255.0, &mut rng)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dither_single,
    bench_dither_trait_methods,
    bench_dither_iter,
    bench_different_types
);
criterion_main!(benches);
