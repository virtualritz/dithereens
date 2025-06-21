use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dithereens::{simple_dither, simple_dither_iter, simple_dither_slice};
use rand::{SeedableRng, rngs::StdRng};
use std::hint::black_box;

// Test RNG that satisfies Rayon trait bounds for benchmarking
#[derive(Clone)]
struct BenchRng {
    inner: StdRng,
}

impl BenchRng {
    fn new(seed: u64) -> Self {
        Self {
            inner: StdRng::seed_from_u64(seed),
        }
    }
}

impl rand::RngCore for BenchRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest)
    }
}

fn bench_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_vs_parallel");

    for size in [100, 1000, 10000, 100000].iter() {
        let values: Vec<f32> =
            (0..*size).map(|i| i as f32 / *size as f32).collect();

        // Sequential: individual calls
        group.bench_with_input(
            BenchmarkId::new("sequential_individual", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    let results: Vec<f32> = values
                        .iter()
                        .map(|&v| simple_dither(black_box(v), 255.0, &mut rng))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Sequential: batch function (when rayon feature is disabled)
        #[cfg(not(feature = "rayon"))]
        group.bench_with_input(
            BenchmarkId::new("sequential_batch", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    let results = simple_dither_iter(
                        black_box(values.clone()),
                        255.0,
                        &mut rng,
                    );
                    black_box(results)
                })
            },
        );

        // Parallel: batch function (when rayon feature is enabled)
        #[cfg(feature = "rayon")]
        group.bench_with_input(
            BenchmarkId::new("parallel_batch", size),
            size,
            |b, _| {
                b.iter(|| {
                    let rng = BenchRng::new(42);
                    let results = simple_dither_iter(
                        black_box(values.clone()),
                        255.0,
                        &rng,
                    );
                    black_box(results)
                })
            },
        );

        // Sequential: in-place
        #[cfg(not(feature = "rayon"))]
        group.bench_with_input(
            BenchmarkId::new("sequential_inplace", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
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

        // Parallel: in-place
        #[cfg(feature = "rayon")]
        group.bench_with_input(
            BenchmarkId::new("parallel_inplace", size),
            size,
            |b, _| {
                b.iter(|| {
                    let rng = BenchRng::new(42);
                    let mut values_copy = values.clone();
                    simple_dither_slice(
                        black_box(&mut values_copy),
                        255.0,
                        &rng,
                    );
                    black_box(values_copy)
                })
            },
        );
    }

    group.finish();
}

fn bench_scaling_with_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("rayon_scaling");

    // Test different data sizes to show how Rayon scales
    for size in [1000, 5000, 10000, 50000, 100000].iter() {
        let values: Vec<f32> =
            (0..*size).map(|i| i as f32 / *size as f32).collect();

        #[cfg(feature = "rayon")]
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            size,
            |b, _| {
                b.iter(|| {
                    let rng = BenchRng::new(42);
                    let results = simple_dither_iter(
                        black_box(values.clone()),
                        255.0,
                        &rng,
                    );
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

fn bench_dither_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("dither_variants");
    let size = 10000;
    let values: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();

    #[cfg(feature = "rayon")]
    {
        let rng = BenchRng::new(42);

        // Full dither function with parallel processing
        group.bench_function("parallel_full_dither", |b| {
            b.iter(|| {
                let rng = rng.clone();
                let results = dithereens::dither_iter(
                    black_box(values.clone()),
                    0.0,
                    255.0,
                    0.5,
                    &rng,
                );
                black_box(results)
            })
        });

        // Simple dither function with parallel processing
        group.bench_function("parallel_simple_dither", |b| {
            b.iter(|| {
                let rng = rng.clone();
                let results =
                    simple_dither_iter(black_box(values.clone()), 255.0, &rng);
                black_box(results)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sequential_vs_parallel,
    bench_scaling_with_size,
    bench_dither_variants
);
criterion_main!(benches);
