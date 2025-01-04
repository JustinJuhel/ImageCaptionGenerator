import os, json
from PIL import Image
from sklearn.model_selection import train_test_split

"""
This function splits the dataset into three parts which proportions depend on the arguments.
Args:
image_arrays: dict; a dict containing (image_name: image np.array) couples
image_captions: dict; a dict containing (image_name: captions list) couples
save_folderpath=None; a folderpath to save the split dataset (optional)
train_size: float=0.8; size of the train subset (in comparison with the original dataset)
val_size: float=0.1; size of the val subset
test_size: float=0.1; size of the test subset
"""


def split_and_save_data(
    image_arrays: dict,
    image_captions: dict,
    save_folderpath=None,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
):
    image_filenames = list(
        image_arrays.keys()
    )  # splitting the dataset into train, validation, and test sets
    train_filenames, temp_filenames = train_test_split(
        image_filenames, test_size=1 - train_size, random_state=42
    )
    val_filenames, test_filenames = train_test_split(
        temp_filenames, test_size=test_size / (test_size + val_size), random_state=42
    )

    train_data = {filename: image_arrays[filename] for filename in train_filenames}
    val_data = {filename: image_arrays[filename] for filename in val_filenames}
    test_data = {filename: image_arrays[filename] for filename in test_filenames}

    train_captions = {
        filename: image_captions[filename] for filename in train_filenames
    }
    val_captions = {filename: image_captions[filename] for filename in val_filenames}
    test_captions = {filename: image_captions[filename] for filename in test_filenames}

    if save_folderpath:  # Save the dataset if save_folderpath is specified
        os.makedirs(save_folderpath, exist_ok=True)
        # saving train data
        train_folder = os.path.join(save_folderpath, "train")
        os.makedirs(train_folder, exist_ok=True)
        for filename, image_array in train_data.items():
            image = Image.fromarray(image_array)
            image.save(os.path.join(train_folder, filename))
        with open(os.path.join(train_folder, "image_captions.json"), "w") as f:
            json.dump(train_captions, f)
        # saving validation data
        val_folder = os.path.join(save_folderpath, "val")
        os.makedirs(val_folder, exist_ok=True)
        for filename, image_array in val_data.items():
            image = Image.fromarray(image_array)
            image.save(os.path.join(val_folder, filename))
        with open(os.path.join(val_folder, "image_captions.json"), "w") as f:
            json.dump(val_captions, f)
        # saving test data
        test_folder = os.path.join(save_folderpath, "test")
        os.makedirs(test_folder, exist_ok=True)
        for filename, image_array in test_data.items():
            image = Image.fromarray(image_array)
            image.save(os.path.join(test_folder, filename))
        with open(os.path.join(test_folder, "image_captions.json"), "w") as f:
            json.dump(test_captions, f)

    return train_data, val_data, test_data, train_captions, val_captions, test_captions


################################################################################################
