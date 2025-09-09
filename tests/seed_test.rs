use dithereens::*;

#[test]
fn test_seed_variation() {
    let values = vec![0.2, 0.5, 0.8];

    println!("\nTesting different seeds with simple_dither_iter:");

    for seed in [42, 43, 123, 200, 1000] {
        let results = simple_dither_iter(values.clone(), 255.0, seed);
        println!("Seed {}: {:?}", seed, results);
    }

    // Test individual values
    println!("\nTesting individual values:");
    for (i, &v) in values.iter().enumerate() {
        println!("\nValue {} at index {}:", v, i);
        for seed in [42, 43, 123, 200, 1000] {
            let result = simple_dither(v, 255.0, i as u32, seed);
            println!("  Seed {}: {}", seed, result);
        }
    }
}
