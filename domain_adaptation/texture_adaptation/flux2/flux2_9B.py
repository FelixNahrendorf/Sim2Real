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
single_input_path = "/app/felix/data/neuralremaster/flux1.1/input/poster/Town10HD_spawnpoint_1/poster_16_images_each/poster_small_01_of_07.png"
#single_input_path = "/app/felix/data/neuralremaster/flux1.1/input/poster/Town01_spawnpoint_1/poster_1024x1024.png"
#single_input_path = "/app/felix/data/neuralremaster/flux1.1/input/poster/Town01_spawnpoint_1/poster_3008x3008.png"
#single_input_path = "/app/felix/data/neuralremaster/flux1.1/input/mixed_domain_poster_double_1024x1024.png"
first_ref_path = "/app/datasets/nuscenes_full/samples/CAM_FRONT/n015-2018-08-01-16-54-05+0800__CAM_FRONT__1533113750662460.jpg"
second_ref_path = "/app/datasets/nuscenes_full/samples/CAM_FRONT/n008-2018-08-31-11-19-57-0400__CAM_FRONT__1535728965912404.jpg"
third_ref_path = "/app/datasets/nuscenes_full/samples/CAM_FRONT/n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800744912460.jpg"
fourth_ref_path = "/app/datasets/nuscenes_full/samples/CAM_FRONT/n015-2018-08-01-15-10-21+0800__CAM_FRONT__1533107700012460.jpg"
fifth_ref_path = "/app/datasets/nuscenes_full/samples/CAM_FRONT/n015-2018-08-01-16-54-05+0800__CAM_FRONT__1533113671762460.jpg"
sixt_ref_path = "/app/datasets/nuscenes_full/samples/CAM_FRONT/n008-2018-08-31-11-19-57-0400__CAM_FRONT__1535729233412404.jpg"

'''prompt = """Transform to photorealistic. Based on professional street photography style:
Road: dark gray worn asphalt with white lane markings slightly faded
Buildings: grey with weathering facades
Vehicles: modern car paint with clear coat reflections
Lighting: afternoon daylight with soft shadows, cloudy sky conditions
Camera: professional automotive surround-view system, slight grain
Preserve exact 16-view grid structure."""'''

prompt = "photorealistic street scene with very smooth cloudy conditions, clear sight, high quality photograph, stay very close to the input image contentwise, keep the grid structure of the image"
#prompt = "dashcam footage street scene with flat gray cloudy lighting, unedited consumer-grade camera, stay close to the input image contentwise"

os.makedirs("./outputs/flux2_9B/poster_16_images_each_town10HD_spawn1", exist_ok=True) 
#os.makedirs("./outputs/flux2_9B/full_106_poster", exist_ok=True) 

# Get list of files to process
if PROCESS_ALL_FILES:
    input_dir = Path(single_input_path).parent
    # Get all PNG files in the directory, sorted
    input_files = sorted(input_dir.glob("poster_small_*.png"))
    #input_files = sorted(input_dir.glob("poster_1024*.png"))
    #input_files = sorted(input_dir.glob("poster_3008*.png")) 
    #input_files = sorted(input_dir.glob("mixed_domain_poster_double_1024*.png"))
    print(f"Found {len(input_files)} files to process")
else:
    input_files = [Path(single_input_path)]
    print("Processing single file")

# Process each file
for input_path in input_files:
    print(f"\nProcessing: {input_path.name}")
    
    # Load input image
    input_img = Image.open(input_path)
    ref_img1 = Image.open(first_ref_path)
    ref_img2 = Image.open(second_ref_path)
    ref_img3 = Image.open(third_ref_path)
    ref_img4 = Image.open(fourth_ref_path)
    ref_img5 = Image.open(fifth_ref_path)
    ref_img6 = Image.open(sixt_ref_path)
    
    # Resize to match output dimensions if needed
    input_img = input_img.resize((1024, 1024))
    ref_img1 = ref_img1.resize((1024, 1024))
    ref_img2 = ref_img2.resize((1024, 1024))
    ref_img3 = ref_img3.resize((1024, 1024))
    ref_img4 = ref_img4.resize((1024, 1024))
    ref_img5 = ref_img5.resize((1024, 1024))
    ref_img6 = ref_img6.resize((1024, 1024))
    # Generate output
    image = pipe(
        prompt=prompt,
        image=[input_img], #+ [ref_img1],# ref_img2, ref_img3, ref_img4, ref_img5, ref_img6],  # This enables image-to-image mode
        height=1024,
        width=1024,
        num_inference_steps=50,
        #guidance_scale=guidance_scale, # "Guidance scale is ignored for step-wise distilled models."
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]
    
    # Create output filename based on input filename
    output_filename = f"flux2_9B_{input_path.stem}_prompt10_inf50.png"
    output_path = os.path.join("./outputs/flux2_9B/poster_16_images_each_town10HD_spawn1", output_filename)
    #output_path = os.path.join("./outputs/flux2_9B/full_106_poster", output_filename)
    
    image.save(output_path)
    print(f"Saved: {output_filename}")

print("\nProcessing complete!")
