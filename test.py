# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to test a CNN model.
    test_cnn - Function used to test a Convolutional Neural Network.
    test_bnn - Function used to test a Bayesian Convolutional Neural Network.
"""


# Built-in/Generic Imports
import os
from argparse import Namespace

# Library Imports
import timm
import torch
import laplace
import numpy as np
from pycm import *
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Own Modules
from utils import log
from model import Classifier
from dataset import get_datasets, Dataset


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


def test_cnn(arguments: Namespace, device: torch.device, test_data: Dataset = None) -> None:
    """
    Function for testing the Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for testing.
    :param device: PyTorch device that will be used for training.
    :param test_data: Dataset object can be passed instead of loading default testing data.
    """

    # Loads the testing data is no test data has been provided.
    if test_data is None:
        data = get_datasets(arguments)
        test_data = data if type(data) == Dataset else data[2]

    # Creates the testing data loader using the dataset objects.
    testing_data_loader = DataLoader(test_data, batch_size=arguments.batch_size * 2,
                                     shuffle=False, num_workers=arguments.data_workers,
                                     pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    if arguments.swin_model:
        # Loads the SWIN Transformer model.
        classifier = timm.create_model("swin_base_patch4_window7_224_in22k", pretrained=False,
                                       num_classes=test_data.num_classes)

    else:
        # Loads the EfficientNet CNN model.
        classifier = Classifier(arguments.efficient_net, test_data.num_classes, pretrained=False)

    # Loads the trained model.
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, f"{arguments.load_model}_best.pt")))

    # Sets the classifier to evaluation mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    # Defines the arrays of predictions, labels and the batch count.
    prediction_list, label_list, batch_count = [], [], 0

    # Loops through the testing data batches with no gradient calculations.
    with torch.no_grad():
        for images, labels in testing_data_loader:
            # Adds to the current batch count.
            batch_count += 1

            # Moves the images to the selected device also appends the labels to the array of labels.
            images = images.to(device)
            label_list += list(labels.cpu().numpy())

            # Performs forward propagation using 16 bit precision.
            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():
                    logits = classifier(images)

            # Performs forward propagation using 32 bit precision.
            else:
                logits = classifier(images)

            # Gets the predictive probabilities and appends them to the array of predictions.
            prediction_list += list(F.softmax(logits, dim=1).cpu().numpy())

            # If the number of batches have been reached end testing.
            if batch_count == arguments.batches_per_epoch:
                break

    # Converts the lists of arrays to NumPy Arrays.
    predictions = np.array(prediction_list)
    labels = np.array(label_list)

    # Gets the labels from the predicted labels from the predictions.
    predictions = np.argmax(predictions, axis=1)

    # Calculates the confusion matrix and testing statistics.
    cm = ConfusionMatrix(labels, predictions, digit=5)

    # Logged the confusion matrix and testing statistics.
    log(arguments, cm)
