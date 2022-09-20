# -*- coding: utf-8 -*-


"""
This file contains the function used to handle loading a configuration file and command line arguments.
    load_configurations - Function to load configurations from a configurations file and command line arguments.
    print_arguments - Function to print the loaded arguments.
"""


# Built-in/Generic Imports
import sys
from configparser import ConfigParser
from argparse import ArgumentParser, Namespace

# Own Models
from utils import log, str_to_bool


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


def load_configurations(description: str) -> Namespace:
    """
    Loads arguments from a configuration file and command line.
    Arguments from the command line override arguments from the configuration file.
    :param description: The description of the application that is shown when using the "--help" command.
    :return: ArgumentParser Namespace object containing the loaded configurations.
    """

    # Creates an ArgumentParser to read the command line arguments.
    argument_parser = ArgumentParser(description=description)

    # Creates a ConfigParser to read configurations file arguments.
    config_parser = ConfigParser()

    # Loads either a specified configurations file or file from the default location.
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config_file":
            config_parser.read(sys.argv[2])
        else:
            config_parser.read("config.ini")
    else:
        config_parser.read("config.ini")

    # Standard Arguments
    argument_parser.add_argument("--config_file", type=str,
                                 default="config.ini",
                                 help="String - File path to the config file.")
    argument_parser.add_argument("--experiment", type=str,
                                 default=config_parser["standard"]["experiment"],
                                 help="String - The name of the current experiment.")
    argument_parser.add_argument("--task", type=str,
                                 default=config_parser["standard"]["task"],
                                 help="String - Task for the application to run.")
    argument_parser.add_argument("--seed", type=int,
                                 default=int(config_parser["standard"]["seed"]),
                                 help="Integer - Seed used to generate random numbers.")

    # Logging Arguments
    argument_parser.add_argument("--verbose", type=str_to_bool,
                                 default=config_parser["logging"]["verbose"].lower() == "true",
                                 help="Boolean - Should outputs should be printed on the terminal.")
    argument_parser.add_argument("--log_dir", type=str,
                                 default=config_parser["logging"]["log_dir"],
                                 help="String - Directory path for where log files are stored.")
    argument_parser.add_argument("--log_interval", type=int,
                                 default=int(config_parser["logging"]["log_interval"]),
                                 help="Integer - .")

    # Dataset Arguments
    argument_parser.add_argument("--dataset", type=str,
                                 default=config_parser["dataset"]["dataset"].lower(),
                                 help="String - Dataset to be used.")
    argument_parser.add_argument("--dataset_dir", type=str,
                                 default=config_parser["dataset"]["dataset_dir"],
                                 help="String - Directory path for where the dataset files are stored.")
    argument_parser.add_argument("--image_x", type=int,
                                 default=int(config_parser["dataset"]["image_x"]),
                                 help="Integer - Width of the image that should be resized to.")
    argument_parser.add_argument("--image_y", type=int,
                                 default=int(config_parser["dataset"]["image_y"]),
                                 help="Integer - Height of the image that should be resized to.")
    argument_parser.add_argument("--val_split", type=float,
                                 default=float(config_parser["dataset"]["val_split"]),
                                 help="Float - Percentage of data to be used for validation.")
    argument_parser.add_argument("--test_split", type=float,
                                 default=float(config_parser["dataset"]["test_split"]),
                                 help="Float - Percentage of data to be used for testing.")
    argument_parser.add_argument("--additional_dataset", type=str,
                                 default=config_parser["dataset"]["additional_dataset"].lower(),
                                 help="String - Additional Dataset to be used for testing.")
    argument_parser.add_argument("--additional_dataset_dir", type=str,
                                 default=config_parser["dataset"]["additional_dataset_dir"],
                                 help="String - Directory path for where the additional dataset files are stored.")

    # Performance Arguments
    argument_parser.add_argument("--data_workers", type=int,
                                 default=int(config_parser["performance"]["data_workers"]),
                                 help="Integer - How many data workers should be used to load the data.")
    argument_parser.add_argument("--use_gpu", type=str_to_bool,
                                 default=config_parser["performance"]["use_gpu"].lower() == "true",
                                 help="Boolean - Should training and testing use GPU acceleration.")
    argument_parser.add_argument("--precision", type=int,
                                 default=int(config_parser["performance"]["precision"]),
                                 help="Integer - The level of precision used by the model.")

    # Model Arguments
    argument_parser.add_argument("--model_dir", type=str,
                                 default=config_parser["model"]["model_dir"],
                                 help="String - Directory path for where the Models are saved.")
    argument_parser.add_argument("--load_model", type=str,
                                 default=config_parser["model"]["load_model"],
                                 help="String - The model to be loaded for testing or fine-tuning.")
    argument_parser.add_argument("--efficient_net", type=int,
                                 default=int(config_parser["model"]["efficient_net"]),
                                 help="Integer - The compound coefficient of the efficient net encoder.")
    argument_parser.add_argument("--swin_model", type=str_to_bool,
                                 default=config_parser["model"]["swin_model"].lower() == "true",
                                 help="Boolean - Should the SWIN model be used instead of EfficientNet.")
    argument_parser.add_argument("--temperature", type=float,
                                 default=float(config_parser["model"]["temperature"]),
                                 help="Float - Temperature parameter used for temperature scaling.")

    # Training Arguments
    argument_parser.add_argument("--epochs", type=int,
                                 default=int(config_parser["training"]["epochs"]),
                                 help="Integer - The number of epochs to be run during training.")
    argument_parser.add_argument("--batch_size", type=int,
                                 default=int(config_parser["training"]["batch_size"]),
                                 help="Integer - The size of the batches used during training (used 2 * for testing).")
    argument_parser.add_argument("--minimum_lr", type=float,
                                 default=float(config_parser["training"]["minimum_lr"]),
                                 help="Float - Value for the minimum learning rate during training.")
    argument_parser.add_argument("--maximum_lr", type=float,
                                 default=float(config_parser["training"]["maximum_lr"]),
                                 help="Float - Value for the maximum learning rate during training.")
    argument_parser.add_argument("--k_folds", type=int,
                                 default=int(config_parser["training"]["k_folds"]),
                                 help="Integer - The number of k folds used for cross validation.")

    # Debug Arguments
    argument_parser.add_argument("--detect_anomaly", type=str_to_bool,
                                 default=config_parser["debug"]["detect_anomaly"].lower() == "true",
                                 help="Boolean - Should Autograd anomaly detection be used.")
    argument_parser.add_argument("--batches_per_epoch", type=int,
                                 default=int(config_parser["debug"]["batches_per_epoch"]),
                                 help="Integer - The number of batches to be run per epoch.")

    # Returns the argument parser.
    return argument_parser.parse_args()


def print_arguments(arguments: Namespace) -> None:
    """
    Prints all arguments in a ArgumentParser Namespace.
    :param arguments: ArgumentParser Namespace object containing arguments.
    """

    # Cycles through all the arguments within the ArgumentParser Namespace.
    for argument in vars(arguments):
        log(arguments, f"{argument: <24}: {getattr(arguments, argument)}")

    # Adds a blank line after printing arguments.
    log(arguments, "\n")
