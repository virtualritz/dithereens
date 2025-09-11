use dithereens::*;

#[test]
fn test_hash_variation() {
    println!("\n=== Testing Hash variation with different seeds ===");

    // Test how much variation we get with different seeds
    let mut differences = Vec::new();

    for index in 0..20 {
        let result_seed42 = dither(0.5_f32, 0.0, 255.0, 0.5, index, 42);
        let result_seed43 = dither(0.5_f32, 0.0, 255.0, 0.5, index, 43);

        let diff = (result_seed42 - result_seed43).abs();
        differences.push(diff);

        println!(
            "Index {}: seed 42 = {:.2}, seed 43 = {:.2}, diff = {:.2}",
            index, result_seed42, result_seed43, diff
        );
    }

    // Count how many are actually different
    let different_count = differences.iter().filter(|&&d| d > 0.01).count();
    println!("\nDifferent results: {}/20", different_count);

    // Calculate average difference
    let avg_diff: f32 =
        differences.iter().sum::<f32>() / differences.len() as f32;
    println!("Average difference: {:.4}", avg_diff);

    // The problem might be that consecutive seeds produce similar results
    // Let's test with more spread out seeds
    println!("\n=== Testing with larger seed differences ===");

    for index in 0..10 {
        let result_seed0 = dither(0.5_f32, 0.0, 255.0, 0.5, index, 0);
        let result_seed100 = dither(0.5_f32, 0.0, 255.0, 0.5, index, 100);
        let result_seed1000 = dither(0.5_f32, 0.0, 255.0, 0.5, index, 1000);

        println!(
            "Index {}: seed 0 = {:.2}, seed 100 = {:.2}, seed 1000 = {:.2}",
            index, result_seed0, result_seed100, result_seed1000
        );
    }
}

#[test]
fn test_different_methods_variation() {
    println!("\n=== Comparing variation across different methods ===");

    // Test R2 method
    println!("\nR2 Method:");
    let r2_42 = R2::new(42);
    let r2_43 = R2::new(43);
    for index in 0..10 {
        let r1 =
            dither_with_linear_rng(0.5_f32, 0.0, 255.0, 0.5, index, &r2_42);
        let r2 =
            dither_with_linear_rng(0.5_f32, 0.0, 255.0, 0.5, index, &r2_43);
        println!(
            "Index {}: seed 42 = {:.2}, seed 43 = {:.2}, diff = {:.2}",
            index,
            r1,
            r2,
            (r1 - r2).abs()
        );
    }

    // Test GoldenRatio method
    println!("\nGoldenRatio Method:");
    let gr_42 = GoldenRatio::new(42);
    let gr_43 = GoldenRatio::new(43);
    for index in 0..10 {
        let r1 =
            dither_with_linear_rng(0.5_f32, 0.0, 255.0, 0.5, index, &gr_42);
        let r2 =
            dither_with_linear_rng(0.5_f32, 0.0, 255.0, 0.5, index, &gr_43);
        println!(
            "Index {}: seed 42 = {:.2}, seed 43 = {:.2}, diff = {:.2}",
            index,
            r1,
            r2,
            (r1 - r2).abs()
        );
    }
}

#[test]
fn test_hash_distribution() {
    println!("\n=== Testing Hash distribution ===");

    // Collect results for many indices with same seed
    let mut results = Vec::new();
    for index in 0..100 {
        let result = dither(0.5_f32, 0.0, 255.0, 0.5, index, 42);
        results.push(result);
    }

    // Check distribution
    let below_127 = results.iter().filter(|&&r| r < 127.5).count();
    let above_127 = results.iter().filter(|&&r| r > 127.5).count();
    let exactly_127_or_128 = results
        .iter()
        .filter(|&&r| r == 127.0 || r == 128.0)
        .count();

    println!("Distribution for 100 samples:");
    println!("  Below 127.5: {}", below_127);
    println!("  Above 127.5: {}", above_127);
    println!("  Exactly 127 or 128: {}", exactly_127_or_128);

    let min = results.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = results.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!("  Range: {:.2} to {:.2}", min, max);
}
