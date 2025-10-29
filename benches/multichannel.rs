use criterion::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};
use dithereens::{InterleavedGradientNoise, simple_dither_slice_2d};

/// Benchmark comparing single-channel vs multi-channel performance.
fn bench_multichannel_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("multichannel_performance");

    for size in [256, 512, 1024].iter() {
        let width = *size;
        let height = *size;
        let method = InterleavedGradientNoise::new(42);

        // Grayscale (1 channel)
        group.bench_with_input(
            BenchmarkId::new("grayscale_1ch", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut pixels = vec![0.5f32; width * height];
                    simple_dither_slice_2d::<1, 0, _, _>(
                        black_box(&mut pixels),
                        width,
                        255.0,
                        &method,
                    );
                    black_box(pixels)
                })
            },
        );

        // RGB with correlated noise (3 channels)
        group.bench_with_input(
            BenchmarkId::new("rgb_correlated_3ch", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut pixels = vec![0.5f32; width * height * 3];
                    simple_dither_slice_2d::<3, 0, _, _>(
                        black_box(&mut pixels),
                        width,
                        255.0,
                        &method,
                    );
                    black_box(pixels)
                })
            },
        );

        // RGB with uncorrelated noise (3 channels)
        group.bench_with_input(
            BenchmarkId::new("rgb_uncorrelated_3ch", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut pixels = vec![0.5f32; width * height * 3];
                    simple_dither_slice_2d::<3, 1, _, _>(
                        black_box(&mut pixels),
                        width,
                        255.0,
                        &method,
                    );
                    black_box(pixels)
                })
            },
        );

        // RGBA with correlated noise (4 channels)
        group.bench_with_input(
            BenchmarkId::new("rgba_correlated_4ch", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut pixels = vec![0.5f32; width * height * 4];
                    simple_dither_slice_2d::<4, 0, _, _>(
                        black_box(&mut pixels),
                        width,
                        255.0,
                        &method,
                    );
                    black_box(pixels)
                })
            },
        );

        // RGBA with uncorrelated noise (4 channels)
        group.bench_with_input(
            BenchmarkId::new("rgba_uncorrelated_4ch", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut pixels = vec![0.5f32; width * height * 4];
                    simple_dither_slice_2d::<4, 1, _, _>(
                        black_box(&mut pixels),
                        width,
                        255.0,
                        &method,
                    );
                    black_box(pixels)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark comparing correlated vs uncorrelated noise overhead.
fn bench_noise_correlation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise_correlation_overhead");

    let width = 512;
    let height = 512;
    let method = InterleavedGradientNoise::new(42);

    // RGB: correlated vs uncorrelated
    group.bench_function("rgb_correlated", |b| {
        b.iter(|| {
            let mut rgb_pixels = vec![0.5f32; width * height * 3];
            simple_dither_slice_2d::<3, 0, _, _>(
                black_box(&mut rgb_pixels),
                width,
                255.0,
                &method,
            );
            black_box(rgb_pixels)
        })
    });

    group.bench_function("rgb_uncorrelated", |b| {
        b.iter(|| {
            let mut rgb_pixels = vec![0.5f32; width * height * 3];
            simple_dither_slice_2d::<3, 1, _, _>(
                black_box(&mut rgb_pixels),
                width,
                255.0,
                &method,
            );
            black_box(rgb_pixels)
        })
    });

    // RGBA: correlated vs uncorrelated
    group.bench_function("rgba_correlated", |b| {
        b.iter(|| {
            let mut rgba_pixels = vec![0.5f32; width * height * 4];
            simple_dither_slice_2d::<4, 0, _, _>(
                black_box(&mut rgba_pixels),
                width,
                255.0,
                &method,
            );
            black_box(rgba_pixels)
        })
    });

    group.bench_function("rgba_uncorrelated", |b| {
        b.iter(|| {
            let mut rgba_pixels = vec![0.5f32; width * height * 4];
            simple_dither_slice_2d::<4, 1, _, _>(
                black_box(&mut rgba_pixels),
                width,
                255.0,
                &method,
            );
            black_box(rgba_pixels)
        })
    });

    group.finish();
}

/// Benchmark scaling with channel count.
fn bench_channel_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("channel_count_scaling");

    let width = 512;
    let height = 512;
    let method = InterleavedGradientNoise::new(42);

    // Test 1, 2, 3, 4 channels with correlated noise
    for channels in [1, 2, 3, 4].iter() {
        group.bench_with_input(
            BenchmarkId::new("correlated", channels),
            channels,
            |b, &ch| {
                b.iter(|| {
                    let mut pixels = vec![0.5f32; width * height * ch];
                    match ch {
                        1 => simple_dither_slice_2d::<1, 0, _, _>(
                            black_box(&mut pixels),
                            width,
                            255.0,
                            &method,
                        ),
                        2 => simple_dither_slice_2d::<2, 0, _, _>(
                            black_box(&mut pixels),
                            width,
                            255.0,
                            &method,
                        ),
                        3 => simple_dither_slice_2d::<3, 0, _, _>(
                            black_box(&mut pixels),
                            width,
                            255.0,
                            &method,
                        ),
                        4 => simple_dither_slice_2d::<4, 0, _, _>(
                            black_box(&mut pixels),
                            width,
                            255.0,
                            &method,
                        ),
                        _ => unreachable!(),
                    }
                    black_box(pixels)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_multichannel_performance,
    bench_noise_correlation_overhead,
    bench_channel_count_scaling
);
criterion_main!(benches);
