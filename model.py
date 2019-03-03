import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import TensorBoard
from keras import utils
from keras import models
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import matplotlib.pyplot as plt
import pickle
from keras.layers.advanced_activations import LeakyReLU
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

        # self.input_data = np.array(self.convert_to_np(input_data))  # if vector (flattened) input
        self.input_data = np.array([np.array(im)[:, :-2] for im in input_data])  # if a matrix (image) input
        self.input_data = np.reshape(self.input_data, (self.n_pieces, self.input_data[0].shape[0],
                                                       self.input_data[0].shape[1], 1))  # if a matrix (image) input
        self.output_notes = self.convert_to_np(output_notes)
        self.output_durations = self.convert_to_np(output_durations)

        # assuming each input image is of the same dimension
        self.input_dim = self.input_data[0].shape
        # assuming each output song if of the same length (number of notes)
        self.output_dim = len(self.output_durations[0])

        # One-hot Encoding the output note data (integer labels of them are arbitrary and meaningless)
        self.output_notes = self.one_hot_encode(self.output_notes)

        # convert to numpy arrays of numpy arrays instead of lists of numpy arrays (better for keras/tf modeling)
        self.output_notes = np.array(self.output_notes)
        self.output_durations = np.array(self.output_durations)

    def one_hot_encode(self, output_notes):
        """
        Given a list of vectors (lists), do a one hot encoding of each element of each vector because the integer label
        of each note is completely arbitrary and has no true numerical value.
        :param output_notes: the output_notes list of lists of notes (integers) where each inner list is a music piece
        :return: a list of lists but the inner lists contain one-hot encodings of the notes
        """
        for i in range(self.n_pieces):
            output_notes[i] = utils.to_categorical(output_notes[i])
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
        :return: a keras model
        """
        print("Input Dimension:", len(self.input_dim))
        if len(self.input_dim) > 1:  # input is a matrix
            the_model = models.Sequential()
            the_model.add(
                Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=self.input_dim, padding='same'))
            the_model.add(LeakyReLU(alpha=0.1))
            the_model.add(MaxPooling2D((2, 2), padding='same'))
            the_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
            the_model.add(LeakyReLU(alpha=0.1))
            the_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
            the_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
            the_model.add(LeakyReLU(alpha=0.1))
            the_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
            the_model.add(Flatten())
            the_model.add(Dense(128, activation='linear'))
            the_model.add(LeakyReLU(alpha=0.1))
            the_model.add(Dense(self.output_durations.shape[1], activation='softmax'))

            the_model.summary()

            the_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                                  metrics=['accuracy'])
            return the_model
        else:  # input is a vector
            # FIXME: not sure if I'll actually use
            input_img = Input(shape=self.input_dim)
            x = Dense(32, activation='relu')(input_img)
            decoded = Dense(self.input_dim[0], activation='sigmoid')(x)
            autoencoder = models.Model(input_img, decoded)
            autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

            return autoencoder

    def train(self, model):
        """
        After building the architecture, train the model on our datasets
        :param model: the keras model architecture
        :return: trained model
        """
        # the model's training progress is being logged: here http://192.168.1.24:6006 /tmp/autoencoder
        print(self.input_data.shape)
        model.fit(self.input_data, self.output_durations, epochs=50, shuffle=True)
        pickle.dump(model, open('autoencoder.pickle', 'wb'))
        return model

    def test(self, a_model):
        """
        Test the convolutional model
        :param a_model: the trained model. Its output could be either a vector of durations or a vector or note values
        """
        test_eval = a_model.evaluate(self.input_data, self.output_notes, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])
