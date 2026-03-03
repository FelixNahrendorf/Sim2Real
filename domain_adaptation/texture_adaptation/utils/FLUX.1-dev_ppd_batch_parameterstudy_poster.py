import os
import sys
import torch
from PIL import Image
from pathlib import Path
import itertools

# Imports - assuming PYTHONPATH is set to project root
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from structured_noise import generate_structured_noise_batch_vectorized

"""
Parameter Analysis Script for FLUX.1-dev on Poster Images
Tests different RADIUS and GUIDANCE_SCALE combinations
Processes 7 poster images (each containing 16 views)

Run from project root:
cd /app/felix/code/NeuralRemaster/PPD-examples
PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_ppd_batch_analysis_posters.py
"""

# ============================================================================
# CONFIGURATION - Poster Images Parameter Study
# ============================================================================

LORA_PATH = "/app/felix/code/NeuralRemaster/PPD-examples/models/ppd/flux1-dev_phipd_lora_302000.safetensors"

# Input directory - poster images
POSTER_INPUT_DIR = "/app/felix/data/neuralremaster/flux1.1/input/poster/Town01_spawnpoint_1/poster_16_images_each"

# Output base directory
BASE_OUTPUT_DIR = "/app/felix/data/neuralremaster/flux1.1/parameter_analysis_posters"

# Parameter ranges for analysis
RADIUS_VALUES = [30, 40, 50, 60, 70, 75, 80, 85, 90, 100, 120, 150, 200]  # 13 values
GUIDANCE_VALUES = [round(0.5 + x * 0.2, 1) for x in range(7)]  # 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7 (7 values)

# ============================================================================
# FIXED PARAMETERS
# ============================================================================

SEED = 42
NUM_INFERENCE_STEPS = 50
HEIGHT = 1024  # Posters are 1024x1024
WIDTH = 1024
LORA_ALPHA = 1.0
DEVICE = "cuda"  # Use CUDA_VISIBLE_DEVICES to control which GPU
TILED_VAE = False

# Enhanced prompt with explicit texture emphasis
UNIVERSAL_PROMPT = """A professional high-resolution photograph of the exact same urban scene captured from a different angle. The vehicle must maintain the same color, make and model across all views - it is the same car. Buildings maintain the same architectural style and consistent appearance.

Street surface: Heavily weathered asphalt with deep cracks, potholes, tire marks, oil stains, aggregate stones visible in worn areas, rough uneven texture, dirt accumulation, realistic road wear patterns, patch repairs with color variations.

Sky: Natural cloudy sky with realistic cloud formations, atmospheric perspective, subtle color gradients, visible cloud texture and detail, natural overcast lighting with depth.

Buildings: Weathered concrete with surface imperfections, stained areas, exposed aggregate, realistic brick with individual brick texture and irregular mortar lines, worn paint with peeling and fading, rust stains and water damage, dirt accumulation in corners.

Vehicle surfaces: Authentic automotive paint with micro-scratches, dust particles, natural reflections showing environment, subtle orange peel texture, panel gaps, realistic tire rubber texture with tread wear.

All surfaces must show realistic material imperfections, natural wear patterns, environmental weathering, fine granular detail, organic irregularity. Shot during overcast conditions with soft diffused natural lighting. Muted natural color palette, photorealistic material quality with high surface detail fidelity. Documentary photography style with visible texture at every scale."""

NEGATIVE_PROMPT = """simulated, rendered, CGI, 3D graphics, video game graphics, CARLA simulation, synthetic data, computer generated, virtual environment, Unity engine, Unreal Engine, artificial lighting, perfect surfaces, pristine clean surfaces, too clean, plastic appearance, smooth flat surfaces, featureless surfaces, uniform textures, no surface detail, flat asphalt, solid color sky, gradient sky without clouds, perfectly clean streets, new pristine materials, cartoon, anime, illustration, digital painting, low quality, blur, low resolution, oversaturated colors, unrealistic materials, overexposed, blown out, overly bright, washed out colors, excessive brightness, harsh sunlight, high key lighting, flat lighting, no texture detail, no surface imperfections, plastic sheen, glossy surfaces, different vehicle colors, inconsistent architecture, mismatched styles, varying appearance across views, smooth roads, perfect geometry"""

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def process_single_image(pipe, input_image_path, output_image_path, radius, guidance_scale):
    """Process a single image through the FLUX pipeline with specific parameters"""
    
    # Load and preprocess input image
    image_in_pil = Image.open(input_image_path).convert("RGB")
    w, h = image_in_pil.size
    
    # Resize to target dimensions
    new_w, new_h = WIDTH, HEIGHT
    image_in_pil = image_in_pil.resize((new_w, new_h), resample=Image.LANCZOS)
    
    # Generate image
    with torch.no_grad():
        # Encode image to latents
        image = pipe.preprocess_image(image_in_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image, tiled=TILED_VAE)

        # Generate structured noise with specified radius
        input_noise = torch.randn_like(input_latents)
        noise = generate_structured_noise_batch_vectorized(input_latents, cutoff_radius=radius, input_noise=input_noise)
        noise = noise.contiguous()

        # Run inference with specified guidance scale
        image = pipe(
            prompt=UNIVERSAL_PROMPT, 
            negative_prompt=NEGATIVE_PROMPT,
            height=new_h, 
            width=new_w,
            cfg_scale=guidance_scale,
            num_inference_steps=NUM_INFERENCE_STEPS,
            noise=noise
        )
        
        # Save output
        image.save(output_image_path)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def collect_poster_images():
    """Collect all poster PNG images from the input directory"""
    if not os.path.exists(POSTER_INPUT_DIR):
        print(f"WARNING: Directory not found: {POSTER_INPUT_DIR}")
        return []
    
    poster_images = []
    for filename in sorted(os.listdir(POSTER_INPUT_DIR)):
        if filename.endswith('.png'):
            poster_images.append(os.path.join(POSTER_INPUT_DIR, filename))
    
    return poster_images

def process_parameter_combination(pipe, image_list, radius, guidance_scale):
    """Process all images with a specific parameter combination"""
    
    # Create output directory with parameter naming - includes "poster" in name
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"spawnpoint_1_poster_rad{radius:03d}_guid{guidance_scale:.1f}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RADIUS={radius}, GUIDANCE_SCALE={guidance_scale}")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(image_list)}")
    
    processed = 0
    errors = 0
    
    for idx, input_image in enumerate(image_list, 1):
        input_filename = Path(input_image).stem
        output_name = os.path.join(output_dir, f"{input_filename}_remastered.jpg")
        
        # Skip if already processed
        if os.path.exists(output_name):
            print(f"  [{idx}/{len(image_list)}] Skipping {input_filename}")
            continue
        
        print(f"  [{idx}/{len(image_list)}] Processing {input_filename}...", end=" ")
        
        try:
            process_single_image(pipe, input_image, output_name, radius, guidance_scale)
            print("SUCCESS")
            processed += 1
        except Exception as e:
            print(f"ERROR: {str(e)}")
            errors += 1
    
    return processed, errors

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("FLUX.1-dev Parameter Analysis - Poster Images")
    print("="*70)
    print(f"Parameter ranges:")
    print(f"  RADIUS: {RADIUS_VALUES}")
    print(f"  GUIDANCE: {GUIDANCE_VALUES[0]} to {GUIDANCE_VALUES[-1]} (step 0.2)")
    print(f"  Total combinations: {len(RADIUS_VALUES) * len(GUIDANCE_VALUES)}")
    print(f"Output base: {BASE_OUTPUT_DIR}")
    print("="*70)
    
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Collect poster images
    poster_images = collect_poster_images()
    
    print(f"\nCollected poster images: {len(poster_images)}")
    for img in poster_images:
        print(f"  - {Path(img).name}")
    
    if len(poster_images) == 0:
        print("\nERROR: No poster images found!")
        return
    
    # Initialize pipeline once
    print("\nInitializing FLUX pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=DEVICE,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )
    
    print(f"Loading LoRA: {LORA_PATH}")
    pipe.load_lora(pipe.dit, LORA_PATH, alpha=LORA_ALPHA)
    print("Pipeline ready!\n")
    
    # Process all parameter combinations
    total_combinations = len(RADIUS_VALUES) * len(GUIDANCE_VALUES)
    combination_count = 0
    total_processed = 0
    total_errors = 0
    
    for radius, guidance in itertools.product(RADIUS_VALUES, GUIDANCE_VALUES):
        combination_count += 1
        print(f"\n{'#'*70}")
        print(f"Combination {combination_count}/{total_combinations}")
        print(f"{'#'*70}")
        
        processed, errors = process_parameter_combination(pipe, poster_images, radius, guidance)
        total_processed += processed
        total_errors += errors
    
    # Final summary
    print("\n" + "="*70)
    print("PARAMETER ANALYSIS COMPLETE")
    print("="*70)
    print(f"Parameter combinations tested: {total_combinations}")
    print(f"Images per combination: {len(poster_images)}")
    print(f"Total images processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"\nResults saved to: {BASE_OUTPUT_DIR}")
    print(f"\nDirectory naming format: spawnpoint_1_poster_radXXX_guidX.X")
    print(f"  Example: spawnpoint_1_poster_rad030_guid0.5")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()