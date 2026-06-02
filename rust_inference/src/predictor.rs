use std::path::Path;
use std::sync::{Arc, Mutex};
use image::{DynamicImage, RgbImage, GrayImage};
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::inputs;
use ort::value::Tensor;

pub struct WoundPredictor {
    session: Arc<Mutex<Session>>,
    image_size: usize,
}

impl WoundPredictor {
    /// Load the ONNX model and configure session options for optimal CPU execution.
    pub fn new<P: AsRef<Path>>(model_path: P, image_size: usize) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([ort::execution_providers::CPU::default().build()])?
            .commit_from_file(model_path)?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            image_size,
        })
    }

    /// Preprocesses a dynamic image:
    /// 1. Resize maintaining aspect ratio so the longest dimension is `image_size`.
    /// 2. Convert to RGB.
    /// 3. Pad into the top-left of a square `[image_size, image_size]` tensor filled with ImageNet mean.
    ///    Note: After normalization, the padding pixels default to `0.0`.
    /// 4. Normalize with ImageNet mean/std.
    /// 
    /// Returns the input tensor, and the scaled height and width `(hs, ws)`.
    pub fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, (u32, u32)) {
        let img_rgb = img.to_rgb8();
        let (w0, h0) = img_rgb.dimensions();
        
        // Calculate aspect-preserving scale factor
        let scale = self.image_size as f32 / (w0.max(h0) as f32);
        let ws = (w0 as f32 * scale).round() as u32;
        let hs = (h0 as f32 * scale).round() as u32;

        // Resize image to active area (bilinear resize)
        let resized = image::imageops::resize(
            &img_rgb, 
            ws, 
            hs, 
            image::imageops::FilterType::Triangle
        );

        // Preallocate tensor [1, 3, image_size, image_size]
        // Setting it to 0.0 initializes the padded region to 0.0,
        // which matches the normalized ImageNet mean.
        let mut tensor = Array4::<f32>::zeros((1, 3, self.image_size, self.image_size));

        // Constants for ImageNet normalization
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        // Fill active area in the tensor
        for y in 0..hs {
            for x in 0..ws {
                let pixel = resized.get_pixel(x, y);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;

                tensor[[0, 0, y as usize, x as usize]] = (r - mean[0]) / std[0];
                tensor[[0, 1, y as usize, x as usize]] = (g - mean[1]) / std[1];
                tensor[[0, 2, y as usize, x as usize]] = (b - mean[2]) / std[2];
            }
        }

        (tensor, (hs, ws))
    }

    /// Run ONNX model inference and retrieve the output probability map.
    pub fn run_inference(&self, input_tensor: Array4<f32>) -> ort::Result<Array4<f32>> {
        // Convert ndarray to Tensor Value
        let input_value = Tensor::from_array(input_tensor)?;
        
        // Lock the session to get mutable access (ort v2 requires &mut self for run)
        let mut session = self.session.lock().map_err(|e| {
            ort::Error::new(format!("Failed to lock session: {}", e))
        })?;
        
        // Run inference
        let outputs = session.run(inputs!["input" => input_value])?;
        
        // Retrieve the output tensor directly by name
        let output_value = outputs.get("output")
            .ok_or_else(|| ort::Error::new("Failed to get model output tensor 'output'"))?;
            
        let output_tensor = output_value.try_extract_tensor::<f32>()?;
        let (_shape, data) = output_tensor;

        // Convert the raw slice to an Array4
        let array = ndarray::Array4::from_shape_vec(
            (1, 1, self.image_size, self.image_size),
            data.to_vec()
        ).map_err(|e| ort::Error::new(format!("Failed to reshape output tensor: {}", e)))?;

        Ok(array)
    }

    /// Postprocesses the raw output tensor:
    /// 1. Crop active region of shape `(hs, ws)` from the top-left.
    /// 2. Apply Sigmoid if values lie outside [0, 1] range.
    /// 3. Resize back to original image shape `(orig_w, orig_h)`.
    /// 4. Apply threshold to get a binary mask (values 1 for wound, 0 for background).
    pub fn postprocess(
        &self, 
        output_tensor: &Array4<f32>, 
        orig_dim: (u32, u32), 
        scaled_dim: (u32, u32), 
        threshold: f32
    ) -> (GrayImage, Array4<f32>) {
        let (orig_w, orig_h) = orig_dim;
        let (hs, ws) = scaled_dim;

        // Extract active area probabilities and map to u8 [0, 255]
        let mut active_prob_img = GrayImage::new(ws, hs);
        
        // We will also return a resized probability map for validation
        let mut prob_crop = Array4::<f32>::zeros((1, 1, hs as usize, ws as usize));

        for y in 0..hs {
            for x in 0..ws {
                let prob_val = output_tensor[[0, 0, y as usize, x as usize]];
                // Apply sigmoid if output contains raw logits
                let prob_val = if prob_val < 0.0 || prob_val > 1.0 {
                    1.0 / (1.0 + (-prob_val).exp())
                } else {
                    prob_val
                };
                
                prob_crop[[0, 0, y as usize, x as usize]] = prob_val;
                let val_u8 = (prob_val * 255.0).clamp(0.0, 255.0) as u8;
                active_prob_img.put_pixel(x, y, image::Luma([val_u8]));
            }
        }

        // Resize the cropped probability map back to original dimensions
        let resized_prob_img = image::imageops::resize(
            &active_prob_img, 
            orig_w, 
            orig_h, 
            image::imageops::FilterType::Triangle
        );

        // Threshold to produce binary mask
        let mut mask = GrayImage::new(orig_w, orig_h);
        let threshold_u8 = (threshold * 255.0).clamp(0.0, 255.0) as u8;

        for y in 0..orig_h {
            for x in 0..orig_w {
                let val = resized_prob_img.get_pixel(x, y)[0];
                let mask_val = if val >= threshold_u8 { 1 } else { 0 };
                mask.put_pixel(x, y, image::Luma([mask_val]));
            }
        }

        (mask, prob_crop)
    }

    /// Blends green color onto the wound region and highlights boundaries with a 2px green border.
    pub fn draw_overlay(&self, img: &DynamicImage, mask: &GrayImage) -> RgbImage {
        let mut overlay = img.to_rgb8();
        let (width, height) = overlay.dimensions();
        let alpha = 0.45_f32;

        // 1. Draw transparent green mask overlay (alpha blend)
        for y in 0..height {
            for x in 0..width {
                if mask.get_pixel(x, y)[0] == 1 {
                    let pixel = overlay.get_pixel_mut(x, y);
                    // R_blend = 0.45 * 0 + 0.55 * R_orig
                    pixel[0] = (pixel[0] as f32 * (1.0 - alpha)) as u8;
                    // G_blend = 0.45 * 255 + 0.55 * G_orig
                    pixel[1] = (alpha * 255.0 + pixel[1] as f32 * (1.0 - alpha)) as u8;
                    // B_blend = 0.45 * 0 + 0.55 * B_orig
                    pixel[2] = (pixel[2] as f32 * (1.0 - alpha)) as u8;
                }
            }
        }

        // 2. Identify boundary pixels (where mask is 1 and neighbors within radius 2 contain 0)
        let mut boundary_pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if mask.get_pixel(x, y)[0] == 1 {
                    let mut is_boundary = false;
                    'outer: for dy in -2..=2 {
                        for dx in -2..=2 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                if mask.get_pixel(nx as u32, ny as u32)[0] == 0 {
                                    is_boundary = true;
                                    break 'outer;
                                }
                            } else {
                                // Image border counts as contour boundary
                                is_boundary = true;
                                break 'outer;
                            }
                        }
                    }
                    if is_boundary {
                        boundary_pixels.push((x, y));
                    }
                }
            }
        }

        // 3. Draw contour outlines as solid green (100% opacity)
        for (x, y) in boundary_pixels {
            let pixel = overlay.get_pixel_mut(x, y);
            pixel[0] = 0;
            pixel[1] = 255;
            pixel[2] = 0;
        }

        overlay
    }
}
