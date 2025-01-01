import os, json, random, textwrap, matplotlib.pyplot as plt, numpy as np
from PIL import Image

"""
This function displays the image of a certain index in an image dictionary
Args:
    image_dict: dict; the dictionary containing the images
    index: int; the wanted index
"""
def visualize_image(image_dict: dict, index: int) -> None:
    filenames = list(image_dict.keys()) # getting the list of filenames (keys) from the dictionary

    if index < 0 or index >= len(filenames): # chceking if the index is within the valid range
        raise IndexError("Index out of range")

    filename = filenames[index] # getting the filename at the specified index
    image_array = image_dict[filename] # getting the image array corresponding to the filename

    # Display the image
    plt.imshow(image_array)
    plt.title(f"Image: {filename}")
    plt.axis('off')  # hidding the axis
    plt.show()

################################################################################################

"""
This function returns the captions corresponding to the image of the chosen index
Args:
    image_captions: dict; a dictionary containing all the captions
    index: int; the wanted index
"""
def get_captions(image_captions: dict, index: int) -> list:
    filenames = list(image_captions.keys()) # getting the list of filenames (keys) from the dictionary

    if index < 0 or index >= len(filenames): # checking if the index is within the valid range
        raise IndexError("Index out of range")

    filename = filenames[index] # getting the filename at the specified index
    captions = image_captions[filename] # getting the list of captions corresponding to the filename
    return captions

################################################################################################

"""
This function displays 9 random images that are stored in an image_folder and their captions.
Args:
    image_folder: str; the path to the folder containing the images
    captions_file: str; the path to the file containing the wanted captions
"""
def explore_dataset(image_folder: str, captions_file: str) -> None:
    captions = {}

    if captions_file.endswith('.txt'): # determining the file type and read the captions file accordingly
        with open(captions_file, 'r') as file:
            for line in file:
                image_name, caption = line.strip().split(',', 1)
                if image_name not in captions:
                    captions[image_name] = []
                captions[image_name].append(caption)
    elif captions_file.endswith('.json'):
        with open(captions_file, 'r') as file:
            captions = json.load(file)
    else:
        raise ValueError("Unsupported captions file format. Please use .txt or .json.")

    # getting a list of all image files in the folder
    image_files = [
        f for f in os.listdir(image_folder) if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        )
    ]

    if len(image_files) < 9: # checking if there are at least 9 images in the folder
        print("Not enough images in the folder to create a 3x3 subplot.")
    else:
        selected_images = random.sample(image_files, 9) # selecting 9 random images
        fig, axes = plt.subplots(3, 3, figsize=(15, 15)) # creating a 3x3 subplot
        axes = axes.flatten()

        for i, image_file in enumerate(selected_images): # displaying the selected images
            image_path = os.path.join(image_folder, image_file)
            with Image.open(image_path) as img:
                width, height = img.size
                axes[i].imshow(img)
                axes[i].set_title(f"Image {i+1}\nSize: {width}x{height}")
                axes[i].axis('off')

                if image_file in captions: # adding captions to the subplot with text wrapping
                    wrapped_captions = [textwrap.fill(caption, width=40) for caption in captions[image_file]]
                    caption_text = "\n".join(wrapped_captions)
                    axes[i].text(0.5, -0.1, caption_text, ha='center', va='top', transform=axes[i].transAxes, fontsize=10, color='blue')

        plt.tight_layout() # adjusting layout
        plt.subplots_adjust(hspace=1.2, bottom=0.2)
        plt.show()

################################################################################################

"""
This function provides some descriptive statistics about the dataset:
- information about the image sizes
- a plot representing the images' widths and heights
Args:
image_folder: str; the path to the folder containing all images
"""
def get_descriptive_statistics(image_folder: str):
    widths, heights = [], [] # lists to store width and height of images
    image_count = 0 # counting the number of images and collect width and height
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(image_folder, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                image_count += 1

    # calculating maximum, minimum, and mean sizes
    max_width = np.max(widths)
    min_width = np.min(widths)
    mean_width = np.mean(widths)

    max_height = np.max(heights)
    min_height = np.min(heights)
    mean_height = np.mean(heights)

    print(f"Number of images: {image_count}") # printing the results
    print(f"Max width: {max_width}, Min width: {min_width}, Mean width: {mean_width}")
    print(f"Max height: {max_height}, Min height: {min_height}, Mean height: {mean_height}")

    plt.scatter(widths, heights, alpha=0.5) # creating a scatter plot
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Scatter Plot of Image Sizes')
    plt.grid(True)
    plt.show()

################################################################################################