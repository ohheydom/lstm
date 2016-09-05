import random

class DocumentScanner:
    """DocumentScanner .... Remember to properly close the file with close_file().

    Parameters
    ----------
    filename : str
        The input file to read from
    sequence_size : int
        Size of the sequences to return via next_sequence()
    shuffle : bool
        If given an alphabetized input, such as baby names, cities, etc..., the model
        will work best if the input file is shuffled and saved as a new file.
    """
    def __init__(self, filename, sequence_size=20, shuffle=False):
        self.f = None
        self.sequence_size = sequence_size

        if shuffle == True:
            filename_shuffled = "{}-shuffled".format(filename)
            self.shuffle_and_save(filename, filename_shuffled)
            self.f = open(filename_shuffled, 'r')
        else:
            self.f = open(filename, 'r')

    def next_sequence(self):
        """next_sequence...

        Returns
        -------
        A string of the sequence
        """
        s = self.f.read(self.sequence_size)

        if len(s) < self.sequence_size:
            self.f.seek(0,0)
        return s

    def close_file(self):
        """close_file properly closes the input file
        """
        self.f.close()

    def shuffle_and_save(self, filename, filename_shuffled):
        """shuffle_and_save ...

        Parameters
        ----------
        doc : str
        filename : str

        Returns
        -------
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            with open(filename_shuffled, 'w') as new_f:
                new_f.writelines(lines)

    def build_character_mappings(self):
        """build_character_mappings...

        Returns
        -------
        d_c_to_i : dict
            Dict that maps characters to indices
        d_i_to_c : dict
            Dict that maps indices to characters
        """
        d_c_to_i = {} #maps character to an index
        d_i_to_c = {} #maps index to character

        for l in self.f:
            for c in l:
                if not c in d_c_to_i:
                    d_c_to_i[c] = len(d_c_to_i)
                    d_i_to_c[len(d_i_to_c)] = c
        self.f.seek(0,0)

        return d_c_to_i, d_i_to_c
