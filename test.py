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
import timm
import torch
import numpy as np
from pycm import *
import pandas as pd
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Own Modules
from utils import log
from dataset import get_datasets, Dataset
from model import CNNClassifier, SWINClassifier


__author__    = ["Jacob Carse", "Tamás Süveges"]
__copyright__ = "Copyright 2022, Dermatology"
__credits__   = ["Jacob Carse", "Tamás Süveges"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse", "Tamás Süveges"]
__email__     = ["j.carse@dundee.ac.uk", "t.suveges@dundee.ac.uk"]
__status__    = "Development"


def test_cnn(arguments: Namespace, device: torch.device, test_data: Dataset = None, fold: int = None) -> None:
    """
    Function for testing the Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for testing.
    :param device: PyTorch device that will be used for training.
    :param test_data: Dataset object can be passed instead of loading default testing data.
    :param fold: Integer for the current k_fold if using cross validation.
    """

    big_time = time.time()

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
        classifier = SWINClassifier(test_data.num_classes)

    else:
        # Loads the EfficientNet CNN model.
        classifier = CNNClassifier(arguments.efficient_net, test_data.num_classes, pretrained=False)

    # Loads the trained model.
    model_name = f"{arguments.experiment}_{'' if fold is None else str(fold) + '_'}best.pt"
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, model_name)))

    # Sets the classifier to evaluation mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    # Defines the arrays of predictions, labels and the batch count.
    prediction_list, label_list, batch_count = [], [], 0

    data_frame = [[] for _ in range(test_data.num_classes + 1)]

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

            # Temperature parameter is applied to the logits.
            logits = torch.div(logits, arguments.temperature)

            predictions = F.softmax(logits, dim=1).cpu().numpy()

            # Adds all information to the dataframe.
            data_frame[0] += labels.tolist()
            for i in range(test_data.num_classes):
                data_frame[1 + i] += predictions[:, i].tolist()

            # Gets the predictive probabilities and appends them to the array of predictions.
            prediction_list += list(predictions)

            # If the number of batches have been reached end testing.
            if batch_count == arguments.batches_per_epoch:
                break

    print(f"{time.time() - big_time}s")

    # Creates the output directory for the output files.
    os.makedirs(arguments.output_dir, exist_ok=True)

    # Creates the DataFrame from the output predictions.
    data_frame = pd.DataFrame(data_frame).transpose()
    data_frame.columns = ["Labels", "MAL", "NV", "BCC", "AK", "BK", "DF", "SCC"]

    # Outputs the output DataFrame to a csv file.
    output_name = f"{arguments.experiment}_{arguments.dataset}{'' if fold is None else '_' + str(fold)}.csv"
    data_frame.to_csv(os.path.join(arguments.output_dir, output_name))

    # Converts the lists of arrays to NumPy Arrays.
    predictions = np.array(prediction_list)
    labels = np.array(label_list)

    # Gets the labels from the predicted labels from the predictions.
    predictions = np.argmax(predictions, axis=1)

    # Calculates the confusion matrix and testing statistics.
    cm = ConfusionMatrix(labels, predictions, digit=5)

    # Logged the confusion matrix and testing statistics.
    log(arguments, cm)
