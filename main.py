import numpy as np
from preprocess import Preprocess


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
        self.sheet_music_path = "/Users/nathanpool/Desktop/Projects4Fun/Music/Datasets/SheetMusic/"
        self.midi_path = "/Users/nathanpool/Desktop/Projects4Fun/Music/Datasets/Midis/"
        # self.sheet_music_path = "/Users/nathanpool/Desktop/AI2/Scales/Sheet Music/"
        # self.midi_path = "/Users/nathanpool/Desktop/AI2/Scales/Midis/"


if __name__ == "__main__":
    main = Main()

    preprocess = Preprocess(main.sheet_music_path, main.midi_path)
    preprocess.load_data()
