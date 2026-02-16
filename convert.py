from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

input_path = "data/test_image/test.heic"
output_path = "data/test_image/test.jpg"

img = Image.open(input_path).convert("RGB")
img.save(output_path, "JPEG", quality=95)

print("Converted test image.")
