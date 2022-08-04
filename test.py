# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to test a CNN model.
    test_cnn - Function used to test a Convolutional Neural Network.
"""


# Built-in/Generic Imports
import os
import time
from argparse import Namespace

# Library Imports
import torch
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

# Own Modules
from utils import log
from model import Classifier
# from dataset import get_datasets


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


def test_cnn(arguments: Namespace, device: torch.device) -> None:
    """
    Function for testing the Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for testing.
    :param device: PyTorch device that will be used for training.
    """

    # Loads the testing data.
    _, _, test_data = (None, None, None) # get_datasets(arguments)

    # Creates the testing data loader using the dataset objects.
    testing_data_loader = DataLoader(test_data, batch_size=arguments.batch_size * 2,
                                     shuffle=False, num_workers=arguments.num_workers,
                                     pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    classifier = Classifier(arguments.efficient_net, test_data.num_classes, pretrained=False)

    # Loads the trained model.
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, f"{arguments.experiment}_best.pt")))

    # Sets the classifier to evaluation mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    batch_count = 0
