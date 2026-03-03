import os
import sys
import torch
from PIL import Image
from pathlib import Path

# Imports - assuming PYTHONPATH is set to project root
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from structured_noise import generate_structured_noise_batch_vectorized

"""
Standalone batch processing script for FLUX.1-dev inference on CARLA SEED4D dataset
Processes spawn_point_1 from Town01 with multi-view consistency

Run from project root:
cd /app/felix/code/NeuralRemaster/PPD-examples
PYTHONPATH=. python examples/flux/model_inference/FLUX.1-dev_ppd_batch.py
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

LORA_PATH = "/app/felix/code/NeuralRemaster/PPD-examples/models/ppd/flux1-dev_phipd_lora_302000.safetensors"

# Input directories
BASE_INPUT_DIR = "/app/felix/data/seed4d/data/data_diverse_1600x900_2poses/static/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle"
NUSCENES_DIR = os.path.join(BASE_INPUT_DIR, "nuscenes_invisible/sensors")
SPHERE_DIR = os.path.join(BASE_INPUT_DIR, "sphere_invisible/sensors")

# Output directories
BASE_OUTPUT_DIR = "/app/felix/data/neuralremaster/flux1.1/spawnpoint_1"
NUSCENES_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "nuscenes_invisible")
SPHERE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "sphere_invisible")

# ============================================================================
# FIXED PARAMETERS FOR MULTI-VIEW CONSISTENCY
# ============================================================================

SEED = 42  # CRITICAL: Fixed seed for consistency across all 106 images
RADIUS = 60
GUIDANCE_SCALE = 2.5
NUM_INFERENCE_STEPS = 50
HEIGHT = 640
WIDTH = 1136
LORA_ALPHA = 1.0
DEVICE = "cuda"
TILED_VAE = False
# Universal prompt that works for all scenarios (street, car, or both)
#UNIVERSAL_PROMPT = """A professional photograph of the exact same urban scene captured from a different angle. The vehicle must maintain the same color, make and model across all views - it is the same car. Buildings maintain the same architectural style and consistent appearance. Translate synthetic textures to authentic photorealistic materials: weathered dark asphalt with realistic cracks and wear patterns, concrete with natural grain and aging, realistic brick textures with mortar detail, authentic vehicle paint with natural reflections and subtle imperfections, metal surfaces with realistic patina. Shot during dark overcast conditions with cloudy lighting. Muted natural color palette, photorealistic material quality with fine surface detail and granularity. Documentary photography style. Properly exposed with preserved shadow detail and highlight information."""
#NEGATIVE_PROMPT = """simulated, rendered, CGI, 3D graphics, video game graphics, CARLA simulation, synthetic data, computer generated, virtual environment, Unity engine, Unreal Engine, artificial lighting, perfect surfaces, too clean, plastic appearance, smooth flat surfaces, uniform textures, cartoon, anime, illustration, digital painting, low quality, blur, low resolution, oversaturated colors, unrealistic materials, overexposed, blown out, overly bright, washed out colors, excessive brightness, harsh sunlight, high key lighting, flat lighting, no texture detail, plastic sheen, different vehicle colors, inconsistent architecture, mismatched styles, varying appearance across views"""# ============================================================================

UNIVERSAL_PROMPT = """A professional high-resolution photograph of the exact same urban scene captured from a different angle. The vehicle must maintain the same color, make and model across all views - it is the same car. Buildings maintain the same architectural style and consistent appearance.

Street surface: Heavily weathered asphalt with deep cracks, potholes, tire marks, oil stains, aggregate stones visible in worn areas, rough uneven texture, dirt accumulation, realistic road wear patterns, patch repairs with color variations.

Sky: Natural cloudy sky with realistic cloud formations, atmospheric perspective, subtle color gradients, visible cloud texture and detail, natural overcast lighting with depth.

Buildings: Weathered concrete with surface imperfections, stained areas, exposed aggregate, realistic brick with individual brick texture and irregular mortar lines, worn paint with peeling and fading, rust stains and water damage, dirt accumulation in corners.

Vehicle surfaces: Authentic automotive paint with micro-scratches, dust particles, natural reflections showing environment, subtle orange peel texture, panel gaps, realistic tire rubber texture with tread wear.

All surfaces must show realistic material imperfections, natural wear patterns, environmental weathering, fine granular detail, organic irregularity. Shot during overcast conditions with soft diffused natural lighting. Muted natural color palette, photorealistic material quality with high surface detail fidelity. Documentary photography style with visible texture at every scale."""

NEGATIVE_PROMPT = """simulated, rendered, CGI, 3D graphics, video game graphics, CARLA simulation, synthetic data, computer generated, virtual environment, Unity engine, Unreal Engine, artificial lighting, perfect surfaces, pristine clean surfaces, too clean, plastic appearance, smooth flat surfaces, featureless surfaces, uniform textures, no surface detail, flat asphalt, solid color sky, gradient sky without clouds, perfectly clean streets, new pristine materials, cartoon, anime, illustration, digital painting, low quality, blur, low resolution, oversaturated colors, unrealistic materials, overexposed, blown out, overly bright, washed out colors, excessive brightness, harsh sunlight, high key lighting, flat lighting, no texture detail, no surface imperfections, plastic sheen, glossy surfaces, different vehicle colors, inconsistent architecture, mismatched styles, varying appearance across views, smooth roads, perfect geometry"""

# INFERENCE FUNCTION
# ============================================================================

def process_single_image(pipe, input_image_path, output_image_path):
    """Process a single image through the FLUX pipeline"""
    
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

        # Generate structured noise
        input_noise = torch.randn_like(input_latents)
        noise = generate_structured_noise_batch_vectorized(input_latents, cutoff_radius=RADIUS, input_noise=input_noise)
        noise = noise.contiguous()

        # Run inference
        image = pipe(
            prompt=UNIVERSAL_PROMPT, 
            negative_prompt=NEGATIVE_PROMPT,
            height=new_h, 
            width=new_w,
            cfg_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            noise=noise
        )
        
        # Save output
        image.save(output_image_path)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def collect_rgb_images(directory):
    """Collect all RGB images from a directory"""
    if not os.path.exists(directory):
        print(f"WARNING: Directory not found: {directory}")
        return []
    
    rgb_images = []
    for filename in sorted(os.listdir(directory)):
        if '_rgb' in filename and filename.endswith('.png'):
            rgb_images.append(os.path.join(directory, filename))
    
    return rgb_images

def process_images(pipe, image_list, output_dir, label):
    """Process a list of images with consistent parameters"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Images to process: {len(image_list)}")
    print(f"Output directory: {output_dir}")
    print(f"Fixed seed: {SEED} (for multi-view consistency)")
    print(f"{'='*70}\n")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, input_image in enumerate(image_list, 1):
        # Get input filename
        input_filename = Path(input_image).stem
        
        # Create output path (keep original filename with _remastered suffix)
        output_name = os.path.join(output_dir, f"{input_filename}_remastered.jpg")
        
        # Skip if already processed
        if os.path.exists(output_name):
            print(f"[{idx}/{len(image_list)}] Skipping {input_filename} (already exists)")
            skipped += 1
            continue
        
        print(f"[{idx}/{len(image_list)}] Processing: {input_filename}")
        
        # Run inference
        try:
            process_single_image(pipe, input_image, output_name)
            print(f"[{idx}/{len(image_list)}] SUCCESS: Saved {output_name}\n")
            processed += 1
        except Exception as e:
            print(f"[{idx}/{len(image_list)}] ERROR processing {input_filename}")
            print(f"    Error: {str(e)}\n")
            errors += 1
            continue
    
    return processed, skipped, errors

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("FLUX.1-dev Batch Processing - spawn_point_1")
    print("="*70)
    print(f"Base input: {BASE_INPUT_DIR}")
    print(f"Base output: {BASE_OUTPUT_DIR}")
    print(f"Fixed seed: {SEED} (ensures multi-view consistency)")
    print("="*70)
    
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed: {SEED}")
    
    # Skip model download - models should already be present
    print("Skipping model download (assuming models are already present)...")
    
    # Initialize pipeline
    print("Initializing FLUX pipeline...")
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
    
    # Load LoRA
    print(f"Loading LoRA from: {LORA_PATH}")
    pipe.load_lora(pipe.dit, LORA_PATH, alpha=LORA_ALPHA)
    print(f"LoRA loaded (alpha={LORA_ALPHA})")
    
    # Collect images from both directories
    nuscenes_images = collect_rgb_images(NUSCENES_DIR)
    sphere_images = collect_rgb_images(SPHERE_DIR)
    
    total_images = len(nuscenes_images) + len(sphere_images)
    
    if total_images == 0:
        print("\nERROR: No RGB images found!")
        return
    
    print(f"\nFound {total_images} total RGB images:")
    print(f"   - NuScenes (ego vehicle views): {len(nuscenes_images)}")
    print(f"   - Sphere (external views): {len(sphere_images)}")
    
    # Process NuScenes images (6 ego vehicle views)
    nuscenes_processed, nuscenes_skipped, nuscenes_errors = process_images(
        pipe,
        nuscenes_images,
        NUSCENES_OUTPUT_DIR,
        "NUSCENES - EGO VEHICLE VIEWS (6 images)"
    )
    
    # Process Sphere images (100 external views)
    sphere_processed, sphere_skipped, sphere_errors = process_images(
        pipe,
        sphere_images,
        SPHERE_OUTPUT_DIR,
        "SPHERE - EXTERNAL VIEWS (100 images)"
    )
    
    # Summary
    total_processed = nuscenes_processed + sphere_processed
    total_skipped = nuscenes_skipped + sphere_skipped
    total_errors = nuscenes_errors + sphere_errors
    
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {total_processed}/{total_images}")
    print(f"Skipped (already exist): {total_skipped}")
    print(f"Errors: {total_errors}")
    print(f"\nOutput directories:")
    print(f"   - NuScenes: {NUSCENES_OUTPUT_DIR}")
    print(f"   - Sphere:   {SPHERE_OUTPUT_DIR}")
    print(f"\nFixed seed {SEED} used for multi-view consistency")
    print(f"\nGeneration parameters:")
    print(f"   - Seed: {SEED}")
    print(f"   - Guidance scale: {GUIDANCE_SCALE}")
    print(f"   - Steps: {NUM_INFERENCE_STEPS}")
    print(f"   - Radius: {RADIUS}")
    print(f"   - LoRA alpha: {LORA_ALPHA}")
    print(f"   - Resolution: {WIDTH}x{HEIGHT}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()