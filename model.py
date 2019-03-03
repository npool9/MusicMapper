import numpy as np
import keras


class Model:
    """
    Given the data that we have received and preprocessed, we will train an autoencoder and possibly other models
    to map between the two modalities (image and midi)
    """

    def __init__(self, input_data, output_notes, output_durations):
        """
        Initialize the input and output (label) data
        :param input_data: a list of PIL images representing pieces of music
        :param output_notes: a list of note lists where each note is an integer and each list is a piece of music
        :param output_durations: a list of durations lists where each float corresponds to a note in a piece of music
        """
        self.n_pieces = len(input_data)

        self.input_data = self.convert_to_np(input_data)
        self.output_notes = self.convert_to_np(output_notes)
        self.output_durations = self.convert_to_np(output_durations)

        # assuming each input image is of the same dimension
        self.input_size = self.input_data[0].shape[0]
        # assuming each output song if of the same length (number of notes)
        self.output_size = len(self.output_durations[0])

        print("One-hot Encoding Note Data...")
        self.output_notes = self.one_hot_encode(self.output_notes)
        print("One-hot Encoding Complete.")

    def one_hot_encode(self, output_notes):
        """
        Given a list of vectors (lists), do a one hot encoding of each element of each vector because the integer label
        of each note is completely arbitrary and has no true numerical value.
        :param output_notes: the output_notes list of lists of notes (integers) where each inner list is a music piece
        :return: a list of lists but the inner lists contain one-hot encodings of the notes
        """
        for i in range(self.n_pieces):
            output_notes[i] = keras.utils.to_categorical(output_notes[i])
        return output_notes

    def convert_to_np(self, data_list):
        """
        Convert a list of lists of data to a list of numpy vectors for training and testing
        :param data_list: a list of lists of data (each inner list is a different training/test example)
        :return: a list of numpy vectors
        """
        for i in range(self.n_pieces):
            if type(data_list[i]) is not list:  # then the input data is image
                data_list[i] = np.array(data_list[i]).flatten()
            else:
                data_list[i] = np.array(data_list[i])
        return data_list

    def build_model(self):
        """
        Define the model architectures for mapping PIL image input to one-hot-encoded note output as well as for mapping
        PIL image input to float vector note duration output.
        """
        inputs = keras.layers.Input(shape=())