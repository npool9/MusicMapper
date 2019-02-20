import numpy as np
from preprocess import Preprocess
import os.path
import pickle

class Main:
    """
    This is an experimental project to evaluate the performance of learning methods for mapping between two modalities:
    paper and performance (i.e. sheet music and a corresponding MIDI file). This class is meant to initialize all the
    sub-tasks of the project.
    """

    def __init__(self):
        """
        Initialize some main parameters such as data paths, etc.
        NOTE: sheet music and midi files are partitioned by composer
        """
        # self.sheet_music_path = "/Users/nathanpool/Desktop/Projects4Fun/Music/Datasets/SheetMusic/"
        # self.midi_path = "/Users/nathanpool/Desktop/Projects4Fun/Music/Datasets/Midis/"
        self.sheet_music_path = "/Users/nathanpool/Desktop/AI2/Scales/Sheet Music/"
        self.midi_path = "/Users/nathanpool/Desktop/AI2/Scales/Midis/"


if __name__ == "__main__":
    main = Main()

    preprocess = Preprocess(main.sheet_music_path, main.midi_path)
    if not os.path.isfile('midi_vectors.pickle') or not os.path.isfile('sheet_matrices.pickle'):
        preprocess.load_data()
        preprocess.get_vectors()
        # get score vectors of notes and save them to a data path
        pickle.dump(preprocess.midis, open('midi_vectors.pickle', 'wb'))
        pickle.dump(preprocess.sheet_music, open('sheet_matrices.pickle', 'wb'))
    else:
        print("Load cached midi vectors")
        preprocess.midis = pickle.load(open('midi_vectors.pickle', 'rb'))
        print("Load cached sheet music")
        preprocess.sheet_music = pickle.load(open('sheet_matrices.pickle', 'rb'))
