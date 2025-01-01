import os
from PIL import Image, ImageOps

"""
This function adds padding if necessary to all images in the dataset,
so that the resulting dataset has an homogeneous size.
Args:
input_folder: str; the path to the input folder (original dataset)
output_folder: str; the path to the folder where you want the new dataset stored
target_size: tuple; the target size of all images
padding_color: tuple=(0, 0, 0); the color of the padding (black by default)
"""
def pad_images(input_folder: str, output_folder: str, target_size: tuple, padding_color: tuple=(0, 0, 0)) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_folder, filename))

            # calculating the padding needed for width and height
            delta_width = target_size[0] - img.size[0]
            delta_height = target_size[1] - img.size[1]
            padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

            # adding padding
            img_padded = ImageOps.expand(img, padding, fill=padding_color)
            img_padded.save(os.path.join(output_folder, filename))

################################################################################################