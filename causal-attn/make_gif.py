import os
import re
import imageio
from pdf2image import convert_from_path

def extract_number(filename):
    match = re.search(r'(\d+)\.(png|pdf)$', filename)
    if match:
        return int(match.group(1))
    return float('inf')

def images_to_gif(directory, gif_name="output.gif", fps=2):
    images = []
    
    files = [f for f in os.listdir(directory) if f.endswith(('.png', '.pdf'))]
    
    files.sort(key=extract_number)
    
    print(f"Found {len(files)} files")
    print("\nFirst few files in order:")
    for f in files[:5]:
        print(f"  {f} (number: {extract_number(f)})")
    
    for file_name in files:
        print(f"Processing: {file_name}")
        file_path = os.path.join(directory, file_name)
        
        if file_name.endswith('.pdf'):
            pages = convert_from_path(file_path)
            for page in pages:
                images.append(page)
        else:
            images.append(imageio.imread(file_path))
    
    duration = 1.0 / fps
    
    out_path = os.path.join(directory, gif_name)
    imageio.mimsave(out_path, images, duration=duration)
    print(f"\nCreated GIF: {out_path}")
    print(f"Duration: {len(images) * duration:.2f} seconds")

def create_gif_from_path(path, fps=2):
    save_name = path.replace("/", "_")
    if not save_name.endswith('.gif'):
        save_name += '.gif'
        
    images_to_gif(path, gif_name=save_name, fps=fps)
    return save_name

if __name__ == "__main__":
    path = "sphere/albert_attention/beta=9.0"
    gif_name = create_gif_from_path(path, fps=2)
    print(f"Created: {gif_name}")