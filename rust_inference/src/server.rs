use std::sync::Arc;
use std::time::Instant;
use std::io::Cursor;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{DynamicImage, RgbImage, GrayImage, GenericImageView};
use serde::Serialize;
use crate::predictor::WoundPredictor;

struct AppState {
    predictor: Arc<WoundPredictor>,
    threshold: f32,
}

#[derive(Serialize)]
struct PredictResponse {
    wound_percentage: f32,
    latency_ms: f64,
    inference_latency_ms: f64,
    mask_base64: String,
    overlay_base64: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model: &'static str,
}

/// Helper to encode RgbImage to Base64 PNG string
fn encode_rgb_png_base64(img: &RgbImage) -> String {
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    let dyn_img = DynamicImage::ImageRgb8(img.clone());
    dyn_img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
    STANDARD.encode(&buffer)
}

/// Helper to encode GrayImage to Base64 PNG string
fn encode_gray_png_base64(img: &GrayImage) -> String {
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    
    // Scale gray mask {0, 1} to {0, 255} before encoding
    let mut scaled_mask = img.clone();
    for pixel in scaled_mask.pixels_mut() {
        if pixel[0] == 1 {
            pixel[0] = 255;
        }
    }
    
    let dyn_img = DynamicImage::ImageLuma8(scaled_mask);
    dyn_img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
    STANDARD.encode(&buffer)
}

/// Simple health check endpoint
async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy",
        model: "wound_seg_efficientnet_b4_unet",
    })
}

/// Predict endpoint: parses uploaded file, runs inference, and returns JSON payload
async fn predict_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut img_bytes = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        if name == "image" {
            if let Ok(bytes) = field.bytes().await {
                img_bytes = Some(bytes);
                break;
            }
        }
    }

    let bytes = match img_bytes {
        Some(b) => b,
        None => return Err((StatusCode::BAD_REQUEST, "Missing 'image' file field".to_string())),
    };

    // Load image from memory
    let start_all = Instant::now();
    let img = match image::load_from_memory(&bytes) {
        Ok(i) => i,
        Err(e) => return Err((StatusCode::BAD_REQUEST, format!("Failed to parse image: {}", e))),
    };

    let orig_dim = img.dimensions();

    // 1. Preprocess
    let (tensor, scaled_hw) = state.predictor.preprocess(&img);

    // 2. Inference
    let start_inf = Instant::now();
    let output = match state.predictor.run_inference(tensor) {
        Ok(out) => out,
        Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, format!("ONNX inference error: {}", e))),
    };
    let inf_latency = start_inf.elapsed().as_secs_f64() * 1000.0;

    // 3. Postprocess
    let (mask, _) = state.predictor.postprocess(&output, orig_dim, scaled_hw, state.threshold);

    // 4. Draw overlay
    let overlay = state.predictor.draw_overlay(&img, &mask);
    let total_latency = start_all.elapsed().as_secs_f64() * 1000.0;

    // Calculate wound percentage
    let total_pixels = mask.width() as f32 * mask.height() as f32;
    let mut wound_pixels = 0_f32;
    for pixel in mask.pixels() {
        if pixel[0] == 1 {
            wound_pixels += 1.0;
        }
    }
    let wound_percentage = (wound_pixels / total_pixels) * 100.0;

    // Convert output images to Base64
    let mask_base64 = encode_gray_png_base64(&mask);
    let overlay_base64 = encode_rgb_png_base64(&overlay);

    Ok(Json(PredictResponse {
        wound_percentage,
        latency_ms: total_latency,
        inference_latency_ms: inf_latency,
        mask_base64,
        overlay_base64,
    }))
}

/// Runs the REST API server on the specified port.
pub async fn start_server(
    predictor: Arc<WoundPredictor>,
    port: u16,
    threshold: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(AppState {
        predictor,
        threshold,
    });

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/predict", post(predict_handler))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("\n==================================================");
    println!("  REST API Server started successfully on port {}", port);
    println!("  Endpoints:");
    println!("    GET  /health");
    println!("    POST /predict  (form-data, 'image' file field)");
    println!("==================================================");
    
    axum::serve(listener, app).await?;
    Ok(())
}
