use std::path::PathBuf;
use std::time::Instant;
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use image::GenericImageView;

mod predictor;
mod benchmark;
mod server;

use predictor::WoundPredictor;
use benchmark::{run_latency_benchmark, run_throughput_benchmark};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum RunMode {
    Predict,
    Batch,
    Benchmark,
    Server,
}

#[derive(Parser, Debug)]
#[command(name = "rust_inference")]
#[command(about = "High-Throughput Low-Latency Rust Inference for Wound Segmentation using ONNX Runtime", long_about = None)]
struct Args {
    /// Running mode
    #[arg(short, long, value_enum, default_value_t = RunMode::Predict)]
    mode: RunMode,

    /// Path to the ONNX model file
    #[arg(short, long, default_value = "checkpoints/wound_seg.onnx")]
    model: PathBuf,

    /// Path to the input image (required for 'predict' and 'benchmark' modes)
    #[arg(short, long)]
    image: Option<PathBuf>,

    /// Path to directory of images (required for 'batch' mode)
    #[arg(short = 'd', long)]
    images_dir: Option<PathBuf>,

    /// Path to output directory to save results
    #[arg(short, long, default_value = "outputs")]
    output_dir: PathBuf,

    /// Probability threshold for segmentation mask
    #[arg(short, long, default_value_t = 0.5)]
    threshold: f32,

    /// Port for the REST API server
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Input image spatial dimension (default 512x512)
    #[arg(long, default_value_t = 512)]
    image_size: usize,

    /// Number of timed iterations in benchmark mode
    #[arg(long, default_value_t = 200)]
    runs: usize,

    /// Number of warmup iterations in benchmark mode
    #[arg(long, default_value_t = 50)]
    warmup: usize,

    /// Thread concurrency count for throughput benchmark
    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    /// Duration (in seconds) for throughput benchmark
    #[arg(long, default_value_t = 5)]
    duration: u64,

    /// Number of threads to parallelize execution within nodes (0 = auto)
    #[arg(long, default_value_t = 0)]
    intra_threads: usize,

    /// Number of threads to parallelize independent nodes (0 = auto)
    #[arg(long, default_value_t = 0)]
    inter_threads: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize ONNX Runtime dynamic library
    let dylib_path = if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        std::path::PathBuf::from(path)
    } else if std::path::Path::new("onnxruntime.dll").exists() {
        std::path::PathBuf::from("onnxruntime.dll")
    } else {
        // Look in the Python virtual environment
        let venv_path = std::path::Path::new("../venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll");
        if venv_path.exists() {
            venv_path.to_path_buf()
        } else {
            // Default fallback path or name
            std::path::PathBuf::from("onnxruntime.dll")
        }
    };

    println!("Initializing ONNX Runtime dynamic library from: {:?}", dylib_path);
    ort::init_from(dylib_path)?
        .with_name("wound_segmentation")
        .commit();

    // 1. Verify model file exists
    if !args.model.exists() {
        eprintln!("Error: ONNX model file not found at {:?}", args.model);
        std::process::exit(1);
    }

    println!("Loading ONNX model from {:?}...", args.model);
    let start_load = Instant::now();
    let predictor = WoundPredictor::new(
        &args.model,
        args.image_size,
        args.intra_threads,
        args.inter_threads,
    )?;
    println!("Model loaded successfully in {:.2} ms.", start_load.elapsed().as_secs_f64() * 1000.0);

    let predictor = std::sync::Arc::new(predictor);

    match args.mode {
        RunMode::Predict => {
            // Predict a single image
            let img_path = match args.image {
                Some(p) => p,
                None => {
                    eprintln!("Error: --image path is required in 'predict' mode.");
                    std::process::exit(1);
                }
            };

            if !img_path.exists() {
                eprintln!("Error: Image file {:?} does not exist.", img_path);
                std::process::exit(1);
            }

            println!("Processing image {:?}...", img_path);
            let img = image::open(&img_path)?;
            let orig_dim = img.dimensions();

            let start_all = Instant::now();
            let (tensor, scaled_hw) = predictor.preprocess(&img);
            let start_inf = Instant::now();
            let output = predictor.run_inference(tensor)?;
            let inf_latency_ms = start_inf.elapsed().as_secs_f64() * 1000.0;
            let (mask, _) = predictor.postprocess(&output, orig_dim, scaled_hw, args.threshold);
            let overlay = predictor.draw_overlay(&img, &mask);
            let total_latency_ms = start_all.elapsed().as_secs_f64() * 1000.0;

            // Calculate wound percentage
            let total_pixels = mask.width() as f32 * mask.height() as f32;
            let mut wound_pixels = 0_f32;
            for pixel in mask.pixels() {
                if pixel[0] == 1 {
                    wound_pixels += 1.0;
                }
            }
            let wound_percentage = (wound_pixels / total_pixels) * 100.0;

            // Scale mask to [0, 255] for saving
            let mut save_mask = mask;
            for p in save_mask.pixels_mut() {
                if p[0] == 1 {
                    p[0] = 255;
                }
            }

            // Create output directory
            std::fs::create_dir_all(&args.output_dir)?;
            let stem = img_path.file_stem().unwrap().to_str().unwrap();
            let mask_path = args.output_dir.join(format!("{}_mask.png", stem));
            let overlay_path = args.output_dir.join(format!("{}_overlay.png", stem));

            save_mask.save(&mask_path)?;
            overlay.save(&overlay_path)?;

            println!("\n--- Single Image Prediction Result ---");
            println!("Output Mask saved to     : {:?}", mask_path);
            println!("Output Overlay saved to  : {:?}", overlay_path);
            println!("Wound Area Percentage    : {:.2}%", wound_percentage);
            println!("Inference Latency        : {:.2} ms", inf_latency_ms);
            println!("End-to-End Latency       : {:.2} ms", total_latency_ms);
            println!("---------------------------------------");
        }

        RunMode::Batch => {
            // Process a directory of images in parallel using Rayon
            let dir_path = match args.images_dir {
                Some(p) => p,
                None => {
                    eprintln!("Error: --images-dir path is required in 'batch' mode.");
                    std::process::exit(1);
                }
            };

            if !dir_path.exists() {
                eprintln!("Error: Directory {:?} does not exist.", dir_path);
                std::process::exit(1);
            }

            let entries = std::fs::read_dir(dir_path)?;
            let mut paths = Vec::new();
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext_str = ext.to_str().unwrap().to_lowercase();
                        if ext_str == "png" || ext_str == "jpg" || ext_str == "jpeg" || ext_str == "bmp" {
                            paths.push(path);
                        }
                    }
                }
            }

            if paths.is_empty() {
                println!("No images found in the specified directory.");
                return Ok(());
            }

            println!("Found {} images. Starting parallel batch inference using Rayon...", paths.len());
            let masks_out = args.output_dir.join("masks");
            let overlays_out = args.output_dir.join("overlays");
            std::fs::create_dir_all(&masks_out)?;
            std::fs::create_dir_all(&overlays_out)?;

            let start_batch = Instant::now();

            paths.par_iter().for_each(|img_path| {
                if let Ok(img) = image::open(img_path) {
                    let orig_dim = img.dimensions();
                    let (tensor, scaled_hw) = predictor.preprocess(&img);
                    if let Ok(output) = predictor.run_inference(tensor) {
                        let (mask, _) = predictor.postprocess(&output, orig_dim, scaled_hw, args.threshold);
                        let overlay = predictor.draw_overlay(&img, &mask);

                        let mut save_mask = mask;
                        for p in save_mask.pixels_mut() {
                            if p[0] == 1 {
                                p[0] = 255;
                            }
                        }

                        let stem = img_path.file_stem().unwrap().to_str().unwrap();
                        let mask_path = masks_out.join(format!("{}_mask.png", stem));
                        let overlay_path = overlays_out.join(format!("{}_overlay.png", stem));

                        let _ = save_mask.save(mask_path);
                        let _ = overlay.save(overlay_path);
                    }
                }
            });

            let elapsed = start_batch.elapsed().as_secs_f64();
            println!("\n--- Batch Inference Result ---");
            println!("Processed {} images in {:.2} seconds.", paths.len(), elapsed);
            println!("Throughput: {:.2} images/sec", paths.len() as f64 / elapsed);
            println!("Masks saved to      : {:?}", masks_out);
            println!("Overlays saved to   : {:?}", overlays_out);
            println!("------------------------------");
        }

        RunMode::Benchmark => {
            // Run latency and throughput benchmarks
            let img_path = match args.image {
                Some(p) => p,
                None => {
                    eprintln!("Error: --image path is required in 'benchmark' mode.");
                    std::process::exit(1);
                }
            };

            if !img_path.exists() {
                eprintln!("Error: Image file {:?} does not exist.", img_path);
                std::process::exit(1);
            }

            let img = image::open(&img_path)?;
            
            // Run single-threaded latency profiling
            run_latency_benchmark(&predictor, &img, args.warmup, args.runs);

            // Run concurrent throughput profiling
            run_throughput_benchmark(std::sync::Arc::clone(&predictor), &img, args.concurrency, args.duration);
        }

        RunMode::Server => {
            // Start the Axum HTTP REST server
            server::start_server(std::sync::Arc::clone(&predictor), args.port, args.threshold).await?;
        }
    }

    Ok(())
}
