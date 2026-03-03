import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import os
from pathlib import Path

# Configuration
PROCESS_ALL_FILES = True
device = "cuda"
dtype = torch.bfloat16

# Load the pipeline
pipe = Flux2KleinPipeline.from_pretrained(
    "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B/snapshots/main",
    torch_dtype=dtype,
    device_map="cuda"
)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Define input path
#input_path = "/app/felix/data/neuralremaster/flux1.1/input/poster/Nuscene_exo_Carla_0/poster_16_images_each/poster_small_01_of_02.png" #16images/poster
#input_path = "/app/felix/data/neuralremaster/flux1.1/input/poster/Nuscene_exo_Carla_0_3ego_6exo/poster/poster_small_01_of_01.png" #9images/poster 3ego-6exo
input_path = "/app/felix/data/neuralremaster/flux1.1/input/poster/Nuscene_exo_Carla_0_6ego_3exo/poster/poster_small_01_of_01.png" #9images/poster 6ego-3exo




#prompt1 = "photorealistic street scene with very smooth cloudy conditions, clear sight, high quality photograph, stay very close to the input image contentwise, keep the grid structure of the image"
#prompt2 = "remaster these grid images of a photorealistic street scene to smooth high resolution, keep the grid structure of the image"
#prompt3 = "remaster these grid images of a photorealistic street scene according to the first six images, close white space, keep the grid structure of the image"
#prompt4 = "remaster these grid images to clear consistent street scene, close white space, keep the grid structure of the image"
#prompt5 = "regenerate these grid images to clear consistent street scene, some of them are views from the top onto the treet, close white space, keep the grid structure of the image"
#prompt6 = "regenerate these grid images to clear street scene, stay very close to the input images, some of them are views from the top onto the street, close white space, keep the grid structure of the image"
#prompt7 = "remaster these grid images to clear street scene very closely to the input images, some of them are views from the top onto the street, close white space, keep the grid structure of the image"
#prompt8 = "remaster these grid images of a street scene very closely to the input images, fill white space, keep the grid structure of the image"
#prompt9 = "remaster these grid images of a street scene very closely to the input images, increase quality and fill white space, keep the grid structure of the image"
#prompt10 = "remaster these grid images of a street scene very closely to the input images, close holes in the street, increase quality and fill white space, keep the grid structure of the image"
#prompt11 = "upscale these grid images of a street scene very closely to the input images, close holes in the street, increase quality and fill white space, keep the grid structure of the image"
#prompt13 = "Photorealistic bird's-eye street view. Complete roads with markings, fill road gaps, stay consistent, preserve grid. 8K sharp focus."
#prompt14 = """High-resolution bird's-eye view synthesis of urban driving scene. 
#Complete partial street renderings: fill missing asphalt with realistic texture and road markings,
#extend incomplete surfaces maintaining perspective consistency, remove white artifacts. 
#Preserve exact grid layout and camera viewpoints. Match lighting and color across all tiles.
#Enhance details: continous lane lines, crosswalks, road surface texture, parked vehicles.
#Photorealistic quality, 8K resolution, architectural visualization standard."""


#prompt15 = """High-resolution bird's-eye view synthesis of urban driving scene grid.
#Each tile represents unique camera viewpoint - preserve individual perspectives and grid structure.
#CRITICAL: Ensure visual consistency between adjacent tiles - matching lighting, color palette, and road texture across the dataset.
#Complete partial street renderings in each tile: fill ALL white artifacts and gaps with realistic asphalt and road markings.
#Close every white space completely with contextually appropriate street content.
#Lane lines should be continuous WITHIN each tile, with realistic perspective.
#Consistent urban scene aesthetics across all tiles while maintaining their individual viewpoints.
#Remove all white star shaped artifacts, no incomplete surfaces, no gaps, fill missing asphalt with realistic texture and road markings.
#Photorealistic 8K quality per tile, architectural visualization standard."""

prompt = """High-resolution bird's-eye view synthesis of urban driving scene grid.
Each tile represents unique camera viewpoint - preserve individual perspectives and grid structure.
CRITICAL: Ensure visual consistency between adjacent tiles - matching lighting, color palette, and road texture across the dataset.
Complete partial street renderings in each tile: fill ALL white artifacts and gaps with realistic asphalt and road markings.
Close every white space completely with contextually appropriate street content.
Lane lines should be continuous WITHIN each tile, with realistic perspective.
Consistent urban scene aesthetics across all tiles while maintaining their individual viewpoints.
Remove all white star shaped artifacts, no incomplete surfaces, no gaps, fill missing asphalt and unclear structures with realistic texture.
Photorealistic 8K quality per tile, architectural visualization standard."""

os.makedirs("./outputs_flux2_9B_remaster/flux2_9B/Nuscene_exo_Carla_0_remaster_3x3_allref", exist_ok=True) 

# Get list of files to process
if PROCESS_ALL_FILES:
    input_dir = Path(input_path).parent
    # Get all PNG files in the directory, sorted
    input_files = sorted(input_dir.glob("poster_small_*.png"))
    print(f"Found {len(input_files)} files to process")
else:
    input_files = [Path(input_path)]
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
        image=[input_img],
        height=1024,
        width=1024,
        num_inference_steps=50,
        #guidance_scale=guidance_scale, # "Guidance scale is ignored for step-wise distilled models."
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]
    
    # Create output filename based on input filename
    output_filename = f"flux2_9B_{input_path.stem}_prompt16_inf50.png"
    output_path = os.path.join("./outputs_flux2_9B_remaster/flux2_9B/Nuscene_exo_Carla_0_remaster_3x3_allref", output_filename)
    
    image.save(output_path)
    print(f"Saved: {output_filename}")

print("\nProcessing complete!")
