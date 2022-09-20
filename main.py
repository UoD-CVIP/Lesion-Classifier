#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
Executable for dermatology experiments.
Can be adjusted using either the configuration file or command line arguments.
"""


# Own Modules Imports
from test import test_cnn
from train import train_cnn
from calibration import find_temperature
from k_fold import k_fold_cross_validation
from utils import log, set_random_seed, get_device
from config import load_configurations, print_arguments


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


if __name__ == "__main__":
    # Loads the arguments from configurations file and command line.
    description = "Experiments on Calibration for Medical Images"
    arguments = load_configurations(description)

    # Displays the loaded arguments.
    log(arguments, "Loaded Arguments:")
    print_arguments(arguments)

    if arguments.seed != -1:
        set_random_seed(arguments.seed)
        log(arguments, f"Set Random Seed to {arguments.seed}")

    # Sets the default device to be used.
    device = get_device(arguments)
    log(arguments, f"Device set to {device}\n")

    # Trains a CNN model.
    if arguments.task.lower() == "train":
        train_cnn(arguments, device)

    # Fine-tunes a CNN model.
    elif arguments.task.lower() == "finetune":
        train_cnn(arguments, device, load_model=True)

    # Tests a CNN model.
    elif arguments.task.lower() == "test":
        test_cnn(arguments, device)

    # Trains a CNN model using k fold validation.
    elif arguments.task.lower() == "train_cv":
        k_fold_cross_validation(arguments, device)

    # Fine-tunes a CNN model using k fold validation.
    elif arguments.task.lower() == "tune_cv":
        k_fold_cross_validation(arguments, device, load_model=True)

    # Finds the temperature used for temperature scaling.
    elif arguments.task.lower() == "temperature":
        find_temperature(arguments, device)

    else:
        log(arguments, "Enter a valid task. \"train\", \"finetune\", \"test\", \"train_cv\" or \"tune_cv\"")
