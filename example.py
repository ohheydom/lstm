import numpy as np
from document_scanner import DocumentScanner
from lstm import LSTM

filename = 'data/male-names' # File name to train the LSTM model on
hidden_size = 100 # Size of the hidden layer of neurons
sequence_size = 100 # Number of steps to unroll the LSTM for

doc_scanner = DocumentScanner(filename, sequence_size, True)
d, d_o = doc_scanner.build_character_mappings()

vocab_size = len(d) 

# These are used for the Adam Optimizer
tups = (0, 0, 0)
adam_params = {'w': tups, 'b': tups, 'ix': tups, 'io': tups, 'fx': tups, 'fo': tups, 'ox': tups, \
        'oo': tups, 'cx': tups, 'co': tups, 'ib': tups, 'cb': tups, 'fb': tups, 'ob': tups}

# Create the LSTM
lstm = LSTM(vocab_size, hidden_size)
o = np.zeros([1, hidden_size])
state = np.zeros([1, hidden_size])

# Train the LSTM and sample every 100 epochs
for i in range(1500):
    X, labels = lstm.vectorize_sequence(doc_scanner.next_sequence(), d)
    loss, state, o, adam_params = lstm.loss_function(X, labels, state, o, adam_params)
    if i % 100 == 0:
        print "Current loss at epoch {}: {}".format(i, loss)
        print "Sample text:"
        x = X[0]
        sample_text = lstm.sample(100, x, state, o)
        print "".join(map(lambda r: d_o[r], sample_text))
        print "\n"
doc_scanner.close_file()
