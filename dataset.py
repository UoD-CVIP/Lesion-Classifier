# -*- coding: utf-8 -*-


"""
The file contains the code used for handling a selected dataset used to train and test the model.
    Dataset - Class for handling the dynamic loading and augmentation of data.
    get_datasets - Function to load the training, validation and testing datasets.
"""


# Built-in/Generic Imports
import os
from argparse import Namespace

# Library Imports
import torch
import numpy as np
import imgaug as ia
import pandas as pd
from pandas import DataFrame
from torch.utils import data
from PIL import Image, ImageFile
from torchvision import transforms
from imgaug import augmenters as iaa


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


class Dataset(data.Dataset):
    """
    Class for handling the dataset used for training and testing.
        init - The initialiser for the class.
        len - Gets the size of the dataset.
        getitem - Gets an individual item from the dataset by index.
        num_classes - Gets the number of classes of the dataset.
        square_image - Method to crop an image to a square.
    """

    def __init__(self, arguments: Namespace, mode: str, df: pd.DataFrame) -> None:
        """
        Initialiser for the class that stores the filenames and labels used to load the images.
        :param arguments: ArgumentParser Namespace containing arguments for loading of the dataset.
        :param mode: String specifying the type of data loaded, "train", "validation" or "test".
        :param df: Pandas DataFrame containing image and label columns.
        """

        # Calls the PyTorch Dataset Initialiser.
        super(Dataset, self).__init__()

        # Stores the arguments and mode in the object.
        self.arguments = arguments
        self.mode = mode

        # Sets the Pillow library to load truncated images.
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # Stores the dataset data frame in the object.
        self.df = df

        # Defines the list of augmentations to be applied to the training images.
        self.augmentation = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Sometimes(0.5, iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                iaa.Sometimes(0.5, iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-45, 45),
                    shear=(-16, 16),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),
                iaa.SomeOf((0, 2),
                   [
                       iaa.Sometimes(0.5,
                                     iaa.OneOf([
                                         iaa.GaussianBlur((0, 3.0)),
                                         iaa.AverageBlur(k=(2, 7)),
                                         iaa.MedianBlur(k=(3, 11))
                                     ])),
                       iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                       iaa.Sometimes(0.5, iaa.Add((-10, 10), per_channel=0.5)),
                       iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5), per_channel=0.5))
                   ], random_order=True
                )
            ], random_order=True
        )

        # Defines the list of augmentations applied to all images.
        self.image_transforms = transforms.Compose([
            transforms.Resize((arguments.image_x, arguments.image_y), transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        """
        Gets the length of the dataset.
        :return: Integer for the length of the dataset.
        """

        return self.df.shape[0]

    def __getitem__(self, index: int) -> (torch.Tensor, int):
        """
        Gets an image and label from the dataset based on inputted index.
        :param index: Integer representing the index of the data from the dataset.
        :return: A PyTorch Tensor with the augmented image and an Integer for the label.
        """

        # Loads the image.
        df_row = self.df.iloc[index]
        image = Image.open(df_row["image"])

        # Augments the image if training.
        if self.mode == "train":
            image = self.augmentation(image=np.asarray(image))
            image = Image.fromarray(image)

        # Crops the image to a square image.
        if image.width != image.height and self.arguments.image_x == self.arguments.image_y:
            image = self.square_image(image)

        # Resizes and normalises the image.
        image = self.image_transforms(image)

        # Returns the image and label.
        return image, df_row["label"]

    @property
    def num_classes(self) -> int:
        """
        Method to return the number of classes in the loaded dataset.
        :return: Integer for the number of classes.
        """

        return 7

    def square_image(self, image: Image) -> Image:
        """
        Method for cropping a given input image into a square.
        :param image: A Pillow Image to be cropped.
        :return: An Square Pillow Image.
        """

        offset = int(abs(image.width - image.height) / 2)
        if image.width > image.height:
            return image.crop([offset, 0, image.width - offset, image.height])
        else:
            return image.crop([0, offset, image.width, image.height - offset])


def get_dataframe(arguments: Namespace) -> DataFrame:
    """
    Gets a DataFrame containing image names and labels of the selected dataset.
    :param arguments: ArgumentParser Namespace containing arguments for dataset loading.
    :return: Pandas DataFrame containing image names and labels.
    """

    # Loads ISIC dataset.
    if arguments.dataset.lower() == "isic":
        # Reads the ISIC dataset csv file containing filenames and labels.
        df = pd.read_csv(os.path.join(arguments.dataset_dir, "ISIC_2019_Training_GroundTruth.csv"))

        # Gets the directory of the ISIC images.
        data_base = os.path.join(arguments.dataset_dir, "ISIC_2019_Training_Input")

        # Removes vascular lesion samples.
        df = df.drop(df[df.VASC == 1].index)

        # Gets the full filenames and labels of the ISIC data.
        filenames = [os.path.join(data_base, x + ".jpg") for x in df["image"].tolist()]
        labels = np.argmax(df.drop(["image", "VASC", "UNK"], 1).to_numpy(), axis=1)

    # Loads SD260 dataset.
    elif arguments.dataset.lower() == "sd260":
        # Reads the SD260 dataset csv file containing filenames and labels.
        df = pd.read_csv(os.path.join(arguments.dataset_dir, "data.csv"))

        # Gets the directory of the SD260 images.
        data_base = os.path.join(arguments.dataset_dir, "data")

        # Gets the full filenames and labels of the SD260 data.
        filenames = [os.path.join(data_base, x + ".jpg") for x in df["image"].tolist()]
        labels = np.argmax(df.drop(["image", "UNK"], 1).to_numpy(), axis=1)

    # Loads NHS Tayside dataset.
    elif arguments.dataset == "tayside":
        # Gets the directory of the SD260 images.
        data_base = os.path.join(arguments.dataset_dir, "data")

    # Loads the NHS Forth Valley dataset.
    elif arguments.dataset == "forth-valley":
        # Gets the directory of the SD260 images.
        data_base = os.path.join(arguments.dataset_dir, "data")

    # Exits script if a valid dataset has not been selected.
    else:
        print("DATASET NOT FOUND: Select either \"ISIC\", \"SD260\", \"Tayside\" or \"Forth-Vally\"")
        quit()

    # Creates a DataFrame with the filenames and labels.
    df = DataFrame([filenames, labels]).transpose()
    df.columns = ["image", "label"]

    # Returns the DataFrame.
    return df


def split_dataframe(df: DataFrame, val_split: float, test_split: float) -> (DataFrame, DataFrame, DataFrame):
    """
    Splits a DataFrame into training, validation and testing DataFrames.
    :param df:
    :param val_split:
    :param test_split:
    :return:
    """

    # Gets the indices of all the data in the dataset.
    indices = np.array(range(df.shape[0]))

    # Shuffles the ISIC dataset.
    random_generator = np.random.default_rng()
    random_generator.shuffle(indices)

    # Split data indices into training, testing and validation sets.
    split_point_1 = int(indices.shape[0] * test_split)
    split_point_2 = int(indices.shape[0] * (val_split + test_split))
    test_indices = indices[0:split_point_1]
    val_indices = indices[split_point_1:split_point_2]
    train_indices = indices[split_point_2::]

    # Creates the DataFrames for each of the data splits.
    train_df = df.take(train_indices)
    val_df = df.take(val_indices)
    test_df = df.take(test_indices)

    # Returns the training, validation and testing DataFrames.
    return train_df, val_df, test_df


def get_datasets(arguments: Namespace) -> (Dataset, Dataset, Dataset):
    """
    Loads the selected dataset and creates Dataset object for training, validation and testing.
    :param arguments: ArgumentParser Namespace containing arguments for data loading.
    :return: Three Dataset objects for training, validation and testing.
    """

    # Gets the DataFrame of image names and labels of the selected dataset.
    df = get_dataframe(arguments)

    # Splits the DataFrame into training, validation and testing DataFrames.
    train_df, val_df, test_df = split_dataframe(df, arguments.val_split, arguments.test_split)

    # Creates the training, validation and testing Dataset objects.
    train_data = Dataset(arguments, "train", train_df)
    val_data = Dataset(arguments, "validation", val_df)
    test_data = Dataset(arguments, "test", test_df)

    # Return the dataset objects.
    return train_data, val_data, test_data
