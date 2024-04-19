from PIL import Image
import sys

# Open the first image to get dimensions
image1 = Image.open('0001.jpg')
width, height = image1.size

num_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 10
# Create a new blank image with width = width * num_frames, height = height
total_width = width * num_frames
new_image = Image.new('RGB', (total_width, height))

# Paste each image into the new image
for i in range(num_frames):
    frame = Image.open(f'{i+1:04d}.jpg')
    new_image.paste(frame, (i * width, 0))

# Save the final image
new_image.save('image_reel.png')
