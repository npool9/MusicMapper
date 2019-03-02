import numpy as np
from PIL import Image
import os
import music21


class Preprocess:
    """
    This class performs all tasks related to data initialization, cleaning, and preprocessing.
    """

    def __init__(self, sheet_music_path, midi_path):
        """
        Initialize the constants and parameters of the preprocessing tasks
        :param sheet_music_path: the direct path to the sheet music dataset
        :param midi_path: the direct path to the midi dataset
        """
        self.sheet_music_path = sheet_music_path
        self.midi_path = midi_path

        # self.composers = ['F. F. Chopin', 'W. A. Mozart', 'C. Debussy']
        self.composers = [""]
        # self.composers = ['C. Debussy']

        self.sheet_music = []
        self.midis = []
        self.midi_lengths = []

    def load_data(self):
        """
        Given the paths of the datasets, load the data into memory by composer
        NOTE: perhaps just stick to one or two composers
        """
        sheet_paths = []
        midi_paths = []
        # first load just the paths into memory so that we can maintain order
        print("Extracting paths...")
        for composer in self.composers:
            sheet_path = self.sheet_music_path + composer + '/'
            for file in os.listdir(sheet_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    sheet_paths.append(sheet_path + file)
            midi_path = self.midi_path + composer + '/'
            for file in os.listdir(midi_path):
                if file.endswith('.mid'):
                    midi_paths.append(midi_path + file)
        print("Sorting path lists...")
        sheet_paths = sorted(sheet_paths)
        midi_paths = sorted(midi_paths)
        print("Path extraction complete.")
        # print("Sheet Music Paths:", sheet_paths)
        # print("Midi Paths:", midi_paths)

        # per shared name among the midis and the sheet music, get EVERY jpeg file associated with that name in sorted
        #  order, load the images, and concatenate them into one so that there is a one-one relationship from sheet
        #  music to midi file.
        global_names = self.find_matches(sheet_paths, midi_paths)
        print("Sheet Paths:", sheet_paths)
        print("Midi Paths:", midi_paths)
        count = 0
        for name in global_names:
            image_group = []
            print("Sheet Music...")
            for sheet in sheet_paths:
                if name in sheet:
                    image_group.append(sheet)
            images = map(Image.open, image_group)
            widths, heights = zip(*(i.size for i in images))
            # keep both of these pairs around just in case we like one more than the other
            total_width = sum(widths)
            max_height = max(heights)
            # max_width = max(widths)
            # total_height = sum(heights)
            # combine all members of the image group to create a score
            score = Image.new('L', (total_width, max_height))
            x_offset = 0
            for im in image_group:
                im = Image.open(im)
                score.paste(im, (x_offset, 0))
                x_offset += im.size[0]
            # display the an example stitched score
            if len(image_group) > 1 and count == 0:
                # score.show()
                count += 1
            # append to the in-memory sheet music dataset
            self.sheet_music.append(score)

            print("Midi...")
            # get corresponding path to the midi file by matching the global name with the full file path (kinda lazy)
            the_midi = None
            for midi in midi_paths:
                if name in midi:
                    the_midi = midi
                    break
            # load the file into the dataset
            midi = music21.converter.parse(the_midi)
            self.midis.append(midi)
            # play a sample midi corresponding to the score that is concatenated and displayed above
            if count == 1:
                # music21.midi.realtime.StreamPlayer(midi).play()
                count += 1
        print("Dataset constructions complete.")

    def find_matches(self, sheet_files, midi_files):
        """
        With our dataset, we have a different JPEG file per page of sheet music, but each MIDI file corresponds to a
        complete piece. So, find matches among the naming schemas between the midi list and the concatenate together
        the JPEG files that correspond to the same (one) piece of music. After this, there should be a one-one
        relationship between midi files and sheet music files.
        :param sheet_files: a list of strings of the direct paths to each sheet music page
        :param midi_files: a list of strings of the direct paths to each midi file
        :return global_names: a list of piece names that have a one-one match between midi and sheet music. With this
         knowledge, we will be able to read every sheet music file corresponding to one piece and concatenate these
         images together.
        """
        midi_names = set()
        sheet_names = set()
        for midi in midi_files:
            # parse out the name of the piece
            slash_index = midi.rfind('/')
            dot_index = midi.rfind('.')
            global_name = midi[slash_index+1:dot_index]
            midi_names.add(global_name)
        for sheet in sheet_files:
            # parse out the name of the piece from the sheet music file, but get rid of 'Page' at the end of the string
            slash_index = sheet.rfind('/')
            # page_index = sheet.rfind('Page')
            page_index = sheet.rfind('.png')
            global_name = sheet[slash_index+1:page_index]
            sheet_names.add(global_name)
        midi_names = sorted(midi_names)
        sheet_names = sorted(sheet_names)
        print("Midi Names:", midi_names)
        print("Sheet Names:", sheet_names)
        notin = [name for name in midi_names if name not in sheet_names] + \
                       [name for name in sheet_names if name not in midi_names]
        for name in notin:
            if name in sheet_names:
                sheet_names.remove(name)
            if name in midi_names:
                midi_names.remove(name)
        print("These lists are now equal:", sheet_names == midi_names)
        return sheet_names

    def get_vectors(self):
        """
        Given the list of midi files that we just created, now convert them to their vector form so that we can properly
        train our models. Simply overwrite self.midis to have a list of vectors instead of a list of midis. These are
        the pitch value vectors for each piece. Include rests.
        """
        for i in range(len(self.midis)):
            notes = []
            note_lengths = []
            midi = self.midis[i]
            try:
                # Given a single stream, partition into a part for each unique instrument
                parts = music21.instrument.partitionByInstrument(midi)
            except:
                pass
            if parts:  # if parts has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, music21.note.Note):
                    # if element is a note, extract pitch
                    notes.append(str(element.pitch))
                    note_lengths.append(float(element.duration.quarterLength))
                elif isinstance(element, music21.chord.Chord):
                    # if element is a chord, append the normal form of the
                    # chord (a list of integers) to the list of notes.
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    note_lengths.append(float(element.duration.quarterLength))
                elif isinstance(element, music21.note.Rest):
                    notes.append("rest")
                    note_lengths.append(float(element.duration.quarterLength))
            self.midis[i] = notes
            self.midi_lengths.append(note_lengths)
            print(note_lengths)

    def note_to_int(self):
        """
        Given each note in self.midis, assign each unique note  to  an integer
        :return:
        """
        full_note_list = set()
        for score in self.midis:
            for note in score:
                full_note_list.add(note)
        full_note_list = sorted(full_note_list)
        return dict((note, number) for number, note in enumerate(full_note_list))