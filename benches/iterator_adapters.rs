use criterion::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};
use dithereens::*;
use rand::{SeedableRng, rngs::SmallRng};

fn bench_iterator_adapters(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterator_adapters");

    for size in [100, 1000, 10000].iter() {
        let values: Vec<f32> =
            (0..*size).map(|i| i as f32 / *size as f32).collect();

        // Benchmark function approach.
        group.bench_with_input(
            BenchmarkId::new("function_simple_dither_iter", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = SmallRng::seed_from_u64(42);
                    let _results = simple_dither_iter(
                        black_box(values.clone()),
                        255.0,
                        &mut rng,
                    );
                });
            },
        );

        // Benchmark iterator adapter approach.
        group.bench_with_input(
            BenchmarkId::new("adapter_simple_dither", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = SmallRng::seed_from_u64(42);
                    let _results: Vec<f32> = black_box(
                        values.iter().copied().simple_dither(255.0, &mut rng),
                    );
                });
            },
        );

        // Benchmark complex iterator chain.
        group.bench_with_input(
            BenchmarkId::new("adapter_complex_chain", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = SmallRng::seed_from_u64(42);
                    let _results: Vec<f32> = black_box(
                        (0..*size)
                            .map(|i| i as f32 / *size as f32)
                            .filter(|&x| x > 0.1)
                            .map(|x| x * 0.8)
                            .simple_dither(255.0, &mut rng),
                    );
                });
            },
        );

        // Benchmark full dither adapter.
        group.bench_with_input(
            BenchmarkId::new("adapter_full_dither", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut rng = SmallRng::seed_from_u64(42);
                    let _results: Vec<f32> = black_box(
                        values
                            .iter()
                            .copied()
                            .dither(0.0, 255.0, 0.5, &mut rng),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_iterator_adapter_vs_manual(c: &mut Criterion) {
    let values: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();

    c.bench_function("manual_loop", |b| {
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut results = Vec::with_capacity(values.len());
            for &value in &values {
                results.push(simple_dither(value, 255.0, &mut rng));
            }
            black_box(results)
        });
    });

    c.bench_function("iterator_adapter", |b| {
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let results: Vec<f32> =
                values.iter().copied().simple_dither(255.0, &mut rng);
            black_box(results)
        });
    });

    c.bench_function("iterator_map_collect", |b| {
        b.iter(|| {
            let mut rng = SmallRng::seed_from_u64(42);
            let results: Vec<f32> = values
                .iter()
                .map(|&value| simple_dither(value, 255.0, &mut rng))
                .collect();
            black_box(results)
        });
    });
}

criterion_group!(
    benches,
    bench_iterator_adapters,
    bench_iterator_adapter_vs_manual
);
criterion_main!(benches);
