import random

class DocumentScanner:
    def __init__(self, doc, sequence_size=20, shuffle=False):
        self.f = None
        self.sequence_size = sequence_size

        if shuffle == True:
            shuffled_doc = "{}-shuffled".format(doc)
            self.shuffle_and_save(doc, shuffled_doc)
            self.f = open(shuffled_doc, 'r')
        else:
            self.f = open(doc, 'r')

    def next_sequence(self):
        s = self.f.read(self.sequence_size)

        if len(s) < self.sequence_size:
            self.f.seek(0,0)
        return s

    def closeFile(self):
        self.f.close()

    def shuffle_and_save(self, doc, filename):
        with open(doc, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            with open(filename, 'w') as new_f:
                new_f.writelines(lines)

    def build_character_mappings(self):
        d_c_to_i = {} #maps character to an index
        d_i_to_c = {} #maps index to character

        for l in self.f:
            for c in l:
                if not c in d_c_to_i:
                    d_c_to_i[c] = len(d_c_to_i)
                    d_i_to_c[len(d_i_to_c)] = c
        self.f.seek(0,0)

        return d_c_to_i, d_i_to_c
