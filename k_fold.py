# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to train and test CNN model using K-fold cross validation.
    k_fold_cross_validation - Function used to cross validate a Convolutional Neural Network.
"""


# Built-in/Generic Imports
from argparse import Namespace

# Library Imports
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Own Modules
from utils import log
from train import train_cnn
from test import test_cnn
from dataset import get_datasets, get_dataframe, Dataset


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


def k_fold_cross_validation(arguments: Namespace, device: torch.device, load_model: bool = False) -> None:
    """
    Trains and tests a CNN model using K-folds cross validation.
    :param arguments: ArgumentParser Namespace containing arguments.
    :param device: PyTorch device that will be used for training.
    :param load_model: Boolean if another trained model should be loaded for fine-tuning.
    """

    # Gets the dataframe containing the filenames and labels for the dataset.
    dataframe = get_dataframe(arguments)

    # Splits the filenames and labels from the dataframe.
    filenames = dataframe["image"].to_numpy()
    labels = dataframe["label"].to_numpy()

    # Edits the arguments to the additional arguments.
    tmp_dataset, tmp_dataset_dir = arguments.dataset, arguments.dataset_dir
    arguments.dataset = arguments.additional_dataset
    arguments.dataset_dir = arguments.additional_dataset_dir

    # Gets the Dataset object for the additional dataset.
    data = get_datasets(arguments)
    additional_dataset = data if type(data) == Dataset else data[2]

    # Resets the arguments.
    arguments.dataset = tmp_dataset
    arguments.dataset_dir = tmp_dataset_dir

    # Creates a KFold iterator.
    k_fold = KFold(n_splits=arguments.k_folds, shuffle=True, random_state=arguments.seed)

    # Loops through the K folds.
    for i, indices in enumerate(k_fold.split(filenames)):
        # Displays the current cross validation fold.
        log(arguments, f"----- Fold {i}\n")

        # Gets the training and testing filenames for the k fold.
        train_indices, test_indices = indices
        train_filenames, test_filenames = filenames[train_indices], filenames[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

        # Creates a DataFrame with the training filenames and labels.
        train_data = pd.DataFrame([train_filenames, train_labels]).transpose()
        train_data.columns = ["image", "label"]

        # Creates a DataFrame with the testing filenames and labels.
        test_data = pd.DataFrame([test_filenames, test_labels]).transpose()
        test_data.columns = ["image", "label"]

        # Gets the indices of all the data in the dataset.
        indices = np.array(range(train_data.shape[0]))

        # Shuffles the dataset indices.
        random_generator = np.random.default_rng()
        random_generator.shuffle(indices)

        # Splits the training data into training and validation.
        split_point = int(indices.shape[0] * arguments.val_split)
        val_indices = indices[0:split_point]
        train_indices = indices[split_point::]

        # Creates the DataFrames for each of the data splits.
        val_data = train_data.take(val_indices)
        train_data = train_data.take(train_indices)

        # Creates the training, validation and testing Dataset objects.
        train_data = Dataset(arguments, "train", train_data)
        val_data = Dataset(arguments, "validation", val_data)
        test_data = Dataset(arguments, "test", test_data)

        # Trains the CNN model using the training and validation data fold.
        arguments.temperature = train_cnn(arguments, device, load_model, train_data, val_data, i)

        # Temporally sets the load model to the newly trained model.
        temp_load_model = arguments.load_model
        arguments.load_model = arguments.experiment

        # Tests the CNN model using the testing data fold.
        test_cnn(arguments, device, test_data, i)

        log(arguments, f"\nAdditional dataset Testing - {arguments.additional_dataset}")

        # Temporally sets the dataset to the additional dataset.
        tmp_dataset = arguments.dataset
        arguments.dataset = arguments.additional_dataset

        # Tests the CNN model using the additional dataset.
        test_cnn(arguments, device, additional_dataset, i)

        # Sets the original load model and dataset.
        arguments.load_model = temp_load_model
        arguments.dataset = tmp_dataset
