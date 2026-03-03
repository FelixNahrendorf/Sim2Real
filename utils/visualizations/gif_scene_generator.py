from PIL import Image
import os

root = "/app/felix/data/pixelsplat_Sim2Real/outputs/scene15_gif_exo-Nuscenes_modelvar_1F_exo30_in192_out192/seed"
output_gif = os.path.join(root, "animation_scene15_gif_exo-Nuscenes_modelvar_1F_exo30_in192_out192.gif")

frames = []
for i in range(39):
    color_dir = os.path.join(root, f"Carla_{i}", "color")
    pngs = [f for f in os.listdir(color_dir) if f.endswith(".png")]
    if not pngs:
        print(f"Warning: No PNG found in Carla_{i}/color, skipping.")
        continue
    img_path = os.path.join(color_dir, pngs[0])
    frames.append(Image.open(img_path).convert("RGB"))
    print(f"Loaded Carla_{i}: {pngs[0]}")

frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],
    duration=200,  # ms per frame
    loop=0,
)
print(f"GIF saved to {output_gif}")