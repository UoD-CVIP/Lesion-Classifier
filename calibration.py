# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to find the temperature parameter for a trained model.
    temperature_nll - Calculates the negative log likelihood for a validation set using a provided temperature.
    optimise_temperature - Finds the temperature for a provided model using
    find_temperature - Loads the validation data and a trained model and finds the temperature parameter.
"""


# Built-in/Generic Imports
import os
from argparse import Namespace

# Library Imports
import timm
import torch
import numpy as np
from scipy import optimize
from torch.cuda import amp
from torch.utils.data import DataLoader

# Own Modules
from utils import log
from model import Classifier
from dataset import get_datasets


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


def temperature_nll(temperature, *args) -> float:
    """
    Calculates the negative log likelihood for the given temperature parameter.
    :param temperature: Floating point value for the temperature.
    :param args: Tuple containing logits and labels.
    :return: Floating point value for the negative log likelihood.
    """

    # Gets the logits and labels from the arguments.
    logits, labels = args

    # divides the logits by the temperature parameter.
    labels = logits / temperature

    # Gets the softmax predictive probabilities from the logits.
    predictions = np.clip(np.exp(logits) / np.sum(np.exp(logits), 1)[:, None], 1e-20, 1. - 1e-20)

    # Returns the negative log likelihood from the softmax predictions and labels.
    return -np.sum(labels * np.log(predictions)) / predictions.shape[0]


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
    logits = torch.cat(logit_list).cpu().numpy()
    labels = torch.cat(label_list).cpu().numpy()

    # Optimises the temperature parameter using L-BFGS-B.
    temperature = optimize.minimize(temperature_nll, 1.0, args=(logits, labels), method="L-BFGS-B",
                                    bounds=((0.05, 5.0),), tol=1e-12).x[0]

    # Returns the temperature value.
    return temperature


def find_temperature(arguments: Namespace, device: torch.device) -> None:
    """
    Loads the validation data and a trained model and finds the temperature parameter.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
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
        classifier = timm.create_model("swin_base_patch4_window7_224_in22k", pretrained=True,
                                       num_classes=val_data.num_classes)

    else:
        # Loads the EfficientNet CNN model.
        classifier = Classifier(arguments.efficient_net, val_data.num_classes)

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
