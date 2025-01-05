import os, json, shutil, kagglehub, pandas as pd, numpy as np
from PIL import Image

"""
This function downloads the Flick-8k dataset from Kaggle to a default location
and then copies it to a chosen destination folder.
Args:
    destination_folder: str="data"; the destination folder
"""


def download_flickr(destination_folder: str = "data"):
    # downloading latest version
    source_folder = kagglehub.dataset_download("adityajn105/flickr8k")
    print("Path to dataset files:", source_folder)

    if os.path.exists(destination_folder):  # checking if the destination folder exists
        # asking the user if they want to overwrite the existing folder
        user_input = input(
            f"The destination folder {destination_folder} already exists. Do you want to overwrite it? (yes/no): "
        )
        if user_input.lower() == "yes":
            shutil.rmtree(destination_folder)  # removing the existing folder
            shutil.copytree(
                source_folder, destination_folder
            )  # copying the source folder to the destination folder
            print(f"The folder {source_folder} has been copied to {destination_folder}")
        else:
            print("Operation cancelled.")
    else:
        shutil.copytree(
            source_folder, destination_folder
        )  # copying the source folder to the destination folder
        print(f"The folder {source_folder} has been copied to {destination_folder}")


################################################################################################

"""
This function downloads the GLoVe dataset from Kaggle
and adds it to the data folder.
"""


def download_glove(destination_folder: str = "data/glove"):
    source_folder = kagglehub.dataset_download(
        "rtatman/glove-global-vectors-for-word-representation"
    )
    print("Path to dataset files:", source_folder)

    if os.path.exists(destination_folder):  # checking if the destination folder exists
        # asking the user if they want to overwrite the existing folder
        user_input = input(
            f"The destination folder {destination_folder} already exists. Do you want to overwrite it? (yes/no): "
        )
        if user_input.lower() == "yes":
            shutil.rmtree(destination_folder)  # removing the existing folder
            shutil.copytree(
                source_folder, destination_folder
            )  # copying the source folder to the destination folder
            print(f"The folder {source_folder} has been copied to {destination_folder}")
        else:
            print("Operation cancelled.")
    else:
        shutil.copytree(
            source_folder, destination_folder
        )  # copying the source folder to the destination folder
        print(f"The folder {source_folder} has been copied to {destination_folder}")


################################################################################################

"""
This function loads the data and returns two dictionaries.
Those two dictionaries are structured as follows:
Keys: image names
Values: np.arrays corresponding to the image or a list of strings corresponding
to the captions attached to the images

Args:
    images_folderpath: str; path to the image folder
    captions_filepath: str; path to the captions.txt file
"""


def load_data(images_folderpath: str, captions_filepath: str):
    image_arrays, image_captions = (
        {},
        {},
    )  # initializing dictionaries to store image arrays and captions
    captions_df = pd.read_csv(captions_filepath)  # reading the captions file

    for (
        index,
        row,
    ) in captions_df.iterrows():  # iterating over the rows in the captions DataFrame
        image_filename = row["image"]
        caption = row["caption"]
        if image_filename not in image_captions:
            image_captions[image_filename] = []
        image_captions[image_filename].append(caption)

    # verifying that all images in the captions file exist in the images folder
    image_files = set(os.listdir(images_folderpath))
    for image_filename in list(image_captions.keys()):
        if image_filename not in image_files:
            del image_captions[image_filename]  # removing entries for missing images
        else:
            # loading the image and convert it to a NumPy array
            image_path = os.path.join(images_folderpath, image_filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            image_arrays[image_filename] = image_array

    return image_arrays, image_captions


################################################################################################

"""
This function load the split dataset, that is structured in three subfolders (train, val, test).
Args:
    dataset_folderpath: str
"""


def load_split_dataset(dataset_folderpath: str):
    # function that loads a subset (train, val, or test) of the dataset.
    def load_subset(subset_name: str):

        subset_images, subset_captions = {}, {}
        subfolder = os.path.join(dataset_folderpath, subset_name)
        if os.path.exists(subfolder):  # checking if the subset folder exists
            # loading captions from the JSON file
            with open(os.path.join(subfolder, "image_captions.json"), "r") as f:
                subset_captions = json.load(f)
            # loading images and convert them to NumPy arrays
            for filename in subset_captions.keys():
                image_path = os.path.join(subfolder, filename)
                image = Image.open(image_path)
                image_array = np.array(image)
                subset_images[filename] = image_array

        return subset_images, subset_captions

    # loading train, validation, and test subsets
    train_data, train_captions = load_subset("train")
    val_data, val_captions = load_subset("val")
    test_data, test_captions = load_subset("test")

    return train_data, val_data, test_data, train_captions, val_captions, test_captions


################################################################################################
