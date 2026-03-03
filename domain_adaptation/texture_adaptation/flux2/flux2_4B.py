import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import os
from pathlib import Path

# Configuration
PROCESS_ALL_FILES = True  # Set to True to process all files, False for single file
device = "cuda"
dtype = torch.bfloat16

# Load the pipeline
pipe = Flux2KleinPipeline.from_pretrained(
    "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/main",
    torch_dtype=dtype,
    device_map="cuda"
)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Define input path
single_input_path = "/app/felix/data/neuralremaster/flux1.1/input/poster/Town01_spawnpoint_1/poster_16_images_each/poster_small_01_of_07.png"

prompt = "photorealistic street scene with natural lighting, high quality photograph, stay close to the input image contentwise"

os.makedirs("./outputs/flux2_4B", exist_ok=True)

# Get list of files to process
if PROCESS_ALL_FILES:
    input_dir = Path(single_input_path).parent
    # Get all PNG files in the directory, sorted
    input_files = sorted(input_dir.glob("poster_small_*.png"))
    print(f"Found {len(input_files)} files to process")
else:
    input_files = [Path(single_input_path)]
    print("Processing single file")

# Process each file
for input_path in input_files:
    print(f"\nProcessing: {input_path.name}")
    
    # Load input image
    input_img = Image.open(input_path)
    
    # Resize to match output dimensions if needed
    input_img = input_img.resize((1024, 1024))
    
    # Generate output
    image = pipe(
        prompt=prompt,
        image=input_img,  # This enables image-to-image mode
        height=1024,
        width=1024,
        num_inference_steps=4,
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]
    
    # Create output filename based on input filename
    output_filename = f"flux2_4B_{input_path.stem}.png"
    output_path = os.path.join("./outputs/flux2_4B/poster_16_images_each", output_filename)
    
    image.save(output_path)
    print(f"Saved: {output_filename}")

print("\nProcessing complete!")