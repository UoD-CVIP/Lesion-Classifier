#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
Executable for dermatology experiments.
Can be adjusted using either the configuration file or command line arguments.
"""


# Own Modules Imports
from utils import *
from config import *


__author__ = ["Jacob Carse", "Tamás Süveges"]
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
