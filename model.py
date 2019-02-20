import numpy as np
from PIL import Image


class Model:
    """
    Given the data that we have received and preprocessed, we will train an autoencoder and possibly other models
    to map between the two modalities (image and midi)
    """

    def __init__(self, input_data, output_data):
        """
        Initialize the input and output (label) data
        """
        self.input_data = input_data
        self.output_data = output_data

