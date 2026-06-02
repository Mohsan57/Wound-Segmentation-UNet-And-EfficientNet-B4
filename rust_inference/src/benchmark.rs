use std::time::{Instant, Duration};
use std::sync::{Arc, Mutex};
use image::{DynamicImage, GenericImageView};
use crate::predictor::WoundPredictor;

/// Performs single-threaded latency profiling and prints P50, P90, P95, and P99 metrics.
pub fn run_latency_benchmark(
    predictor: &WoundPredictor,
    img: &DynamicImage,
    warmup: usize,
    runs: usize,
) -> f64 {
    println!("\n--- End-to-End Latency Benchmark (Single-Threaded) ---");
    println!("Warmup runs: {}, Timed runs: {}", warmup, runs);

    // Warmup
    for i in 0..warmup {
        let (tensor, scaled_hw) = predictor.preprocess(img);
        let output = predictor.run_inference(tensor).unwrap();
        let _ = predictor.postprocess(&output, img.dimensions(), scaled_hw, 0.5);
        if (i + 1) % 10 == 0 {
            println!("  Warmup {}/{} completed", i + 1, warmup);
        }
    }

    let mut end_to_end_latencies = Vec::with_capacity(runs);
    let mut inference_only_latencies = Vec::with_capacity(runs);

    for _ in 0..runs {
        let start_all = Instant::now();
        let (tensor, scaled_hw) = predictor.preprocess(img);
        
        let start_inf = Instant::now();
        let output = predictor.run_inference(tensor).unwrap();
        let inf_duration = start_inf.elapsed().as_secs_f64() * 1000.0; // ms

        let _ = predictor.postprocess(&output, img.dimensions(), scaled_hw, 0.5);
        let total_duration = start_all.elapsed().as_secs_f64() * 1000.0; // ms

        end_to_end_latencies.push(total_duration);
        inference_only_latencies.push(inf_duration);
    }

    // Sort to compute percentiles
    end_to_end_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    inference_only_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_e2e = end_to_end_latencies.iter().sum::<f64>() / runs as f64;
    let mean_inf = inference_only_latencies.iter().sum::<f64>() / runs as f64;

    let p50 = end_to_end_latencies[runs / 2];
    let p90 = end_to_end_latencies[(runs as f64 * 0.90) as usize];
    let p95 = end_to_end_latencies[(runs as f64 * 0.95) as usize];
    let p99 = end_to_end_latencies[(runs as f64 * 0.99) as usize];

    println!("--------------------------------------------------");
    println!("Inference-Only Latency (Mean) : {:.2} ms", mean_inf);
    println!("End-to-End Latency (Mean)     : {:.2} ms", mean_e2e);
    println!("End-to-End P50 (Median)       : {:.2} ms", p50);
    println!("End-to-End P90                : {:.2} ms", p90);
    println!("End-to-End P95                : {:.2} ms", p95);
    println!("End-to-End P99                : {:.2} ms", p99);
    println!("--------------------------------------------------");
    println!("Estimated Single-Thread FPS   : {:.1}", 1000.0 / mean_e2e);
    println!("--------------------------------------------------");

    mean_e2e
}

/// Spawns multiple threads to run inference concurrently and measure peak QPS (Queries Per Second).
pub fn run_throughput_benchmark(
    predictor: Arc<WoundPredictor>,
    img: &DynamicImage,
    concurrency: usize,
    duration_secs: u64,
) {
    println!("\n--- Concurrent Throughput Benchmark ---");
    println!("Concurrency (threads): {}, Duration: {} seconds", concurrency, duration_secs);

    let img_arc = Arc::new(img.clone());
    let total_queries = Arc::new(Mutex::new(0));

    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(duration_secs);

    let mut handles = Vec::new();

    for _ in 0..concurrency {
        let predictor_clone = Arc::clone(&predictor);
        let img_clone = Arc::clone(&img_arc);
        let counter_clone = Arc::clone(&total_queries);

        let handle = std::thread::spawn(move || {
            let mut local_count = 0;
            // Pre-preprocess to avoid scaling inside the timing loop, or include it
            // We want to measure the throughput of the end-to-end pipeline, so let's include it
            while Instant::now() < end_time {
                let (tensor, scaled_hw) = predictor_clone.preprocess(&img_clone);
                if let Ok(output) = predictor_clone.run_inference(tensor) {
                    let _ = predictor_clone.postprocess(&output, img_clone.dimensions(), scaled_hw, 0.5);
                    local_count += 1;
                }
            }

            let mut count = counter_clone.lock().unwrap();
            *count += local_count;
        });

        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }

    let actual_duration = start_time.elapsed().as_secs_f64();
    let total = *total_queries.lock().unwrap();
    let qps = total as f64 / actual_duration;

    println!("--------------------------------------------------");
    println!("Total Queries Completed : {}", total);
    println!("Actual Duration         : {:.2} seconds", actual_duration);
    println!("Throughput (QPS)        : {:.2} queries/sec", qps);
    println!("--------------------------------------------------");
}
