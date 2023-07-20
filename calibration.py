# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to find the temperature parameter for a trained model.
    optimise_temperature - Finds the temperature for a provided model using
    find_temperature - Loads the validation data and a trained model and finds the temperature parameter.
"""


# Built-in/Generic Imports
import os
from argparse import Namespace

# Library Imports
import timm
import torch
from torch.cuda import amp
from torch.optim import LBFGS
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Own Modules
from utils import log
from dataset import get_datasets
from model import CNNClassifier, SWINClassifier


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


def optimise_temperature(arguments: Namespace, classifier: torch.nn.Module, data_loader: DataLoader,
                         device: torch.device) -> float:
    """
    Finds an optimal temperature parameter that minimises negative log likelihood on a given validation set.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param classifier: Pytorch Module of a trained classifier.
    :param data_loader: Pytorch DataLoader for a validation set used to optimise temperature.
    :param device: PyTorch device that will be used for training.
    :return: Floating point value for the output temperature score.
    """

    # Creates the temperature parameter and optimiser for the parameter.
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    temp_optimiser = LBFGS([temperature], lr=0.01, max_iter=1000, line_search_fn="strong_wolfe")

    # Defines the logit and label lists.
    logit_list, label_list = [], []

    # Loops through the data loaded without gradient calculation.
    with torch.no_grad():
        for images, labels in data_loader:

            # Moves the images and labels to the selected device.
            images = images.to(device)
            labels = labels.to(device)

            # Gets the logits from the model using 16 bit precision.
            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():
                    logits = classifier(images)

            # Gets the logtis from the model using 32 bit precision.
            else:
                logits = classifier(images)

            # Appends the logits and labels to the lists of logits and labels.
            logit_list.append(logits)
            label_list.append(labels)

    # Moves the logits and labels from the gpu to the cpu.
    logit_list = torch.cat(logit_list)
    label_list = torch.cat(label_list)

    def _eval() -> torch.Tensor:
        """
        Evaluation function for the temperature scaling optimiser.
        :return: PyTorch Tensor for temperature scaling loss.
        """

        temp_loss = F.cross_entropy(torch.div(logit_list, temperature), label_list)
        temp_loss.backward()
        return temp_loss.item()

    # Performs a step using the temperature scaling optimiser.
    temp_optimiser.step(_eval)

    # Returns the temperature value.
    return temperature.item()


def find_temperature(arguments: Namespace, device: torch.device) -> float:
    """
    Loads the validation data and a trained model and finds the temperature parameter.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
    :return: Temperature value.
    """

    # Loads the validation data.
    _, val_data, _ = get_datasets(arguments)

    # Creates the validation data loader using the dataset object
    validation_data_loader = DataLoader(val_data, batch_size=arguments.batch_size * 2,
                                        shuffle=False, num_workers=arguments.data_workers,
                                        pin_memory=False, drop_last=False)

    log(arguments, "Loaded Dataset\n")

    # Initialises the classifier model.
    if arguments.swin_model:
        # Loads the SWIN Transformer model.
        classifier = SWINClassifier(val_data.num_classes)

    else:
        # Loads the EfficientNet CNN model.
        classifier = CNNClassifier(arguments.efficient_net, val_data.num_classes)

    # Loads a trained model.
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, f"{arguments.experiment}_best.pt")))

    # Sets the classifier to training mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    log(arguments, "Loaded Model\n")

    # Finds the temperature parameter.
    temperature = optimise_temperature(arguments, classifier, validation_data_loader, device)

    log(arguments, f"Temperature = {temperature}")

    return temperature


def rewrite_config(arguments: Namespace, temperature: float) -> None:
    """
    Adds the new temperature value to the config file.
    :param arguments: ArgumentParser Namespace object with arguments used for calibration.
    :param temperature: Floating point value for the new temperature value to rewrite.
    """

    with open(arguments.config_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    lines[31] = f"temperature            = {temperature}\n"

    with open(arguments.config_file, 'w', encoding="utf-8") as file:
        file.writelines(lines)
