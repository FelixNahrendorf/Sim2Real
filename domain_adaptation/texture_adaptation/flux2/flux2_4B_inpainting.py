import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import os

device = "cuda"
dtype = torch.bfloat16

# Load the pipeline
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", 
    torch_dtype=dtype
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Load input image

### domain transfer
#input_img = Image.open("/app/code/share/felix/code/NeuralRemaster/models/ppd/0_rgb_town01_spawn1.png")
#input_img = Image.open("/app/code/share07/felix2/seed4d/data/data_diverse_1600x900_2poses/posters/Town01_spawnpoint_1/poster_16_images_each/poster_small_01_of_07.png")
#input_img = Image.open("/app/code/share07/felix2/seed4d/data/data_diverse_1600x900_2poses/posters/Town01_spawnpoint_1/poster_2048x2048.png")
#input_img = Image.open("/app/code/share/felix/code/NeuralRemaster/models/ppd/000001_Carla_0.png")

### inpainting

input_img = Image.open("/app/code/share/felix/code/Sim2Real/domain_adaptation/texture_adaptation/diffusion_models/FLUX2_4B/inputs/poster_nuscene_ego_exo_with_borders_2048_653.png")

# Resize to match output dimensions if needed
input_img = input_img.resize((672, 2048))

#prompt = "photorealistic street scenes with natural lighting, high quality photographs, stay close to the input image contentwise"
prompt = "match the quality of the grid of twelve images, it is a photorealistic street scene with natural lighting, high quality photograph, stay close to the input image contentwise"

os.makedirs("./outputs", exist_ok=True)

# For image editing, use the 'image' parameter
image = pipe(
    prompt=prompt,
    image=input_img,  # This enables image-to-image mode
    height=672,
    width=2048,
    #guidance_scale=5.0, # not applied to destilled models #Guidance scale for generation. Controls how closely the output follows the prompt. Minimum: 1.5, maximum: 10, default: 4.5.
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]

image.save("./outputs/flux2_4B/flux2_4B_poster_nuscene_ego_exo_with_borders_2048_683.png")


'''import torch
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
    "black-forest-labs/FLUX.2-klein-4B", 
    torch_dtype=dtype
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Define input path
single_input_path = "/app/code/share07/felix2/seed4d/data/data_diverse_1600x900_2poses/posters/Town01_spawnpoint_1/poster_16_images_each/poster_small_01_of_07.png"

prompt = "photorealistic street scene with natural lighting, high quality photograph, stay close to the input image contentwise"

os.makedirs("./outputs", exist_ok=True)

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
    output_filename = f"flux2-klein_{input_path.stem}.png"
    output_path = os.path.join("./outputs/poster_16_images_each", output_filename)
    
    image.save(output_path)
    print(f"Saved: {output_filename}")

print("\nProcessing complete!")'''