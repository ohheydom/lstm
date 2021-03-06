import numpy as np
from document_scanner import DocumentScanner
from lstm import LSTM

# Filename on which to train the LSTM model
filename = 'data/nirvana-lyrics'

# Size of the hidden layer of neurons
hidden_size = 100

# Number of steps for which to unroll the LSTM
sequence_size = 100

doc_scanner = DocumentScanner(filename, sequence_size)
d, d_o = doc_scanner.build_character_mappings()

vocab_size = len(d)

# Create the LSTM
lstm = LSTM(vocab_size, hidden_size)

# Build zero valued matrices of the output and state
h = np.zeros([1, hidden_size])
state = np.zeros([1, hidden_size])

# Build initial Adam parameters that will be used and update during training
adam_params = lstm.build_adam_params()

# Train the LSTM and sample every 100 epochs
for i in range(20000):
    X, labels = lstm.vectorize_sequence(doc_scanner.next_sequence(), d)
    loss, state, h, adam_params = lstm.bptt(X, labels, state, h, adam_params)
    if i % 100 == 0:
        print "Current loss at epoch {}: {}".format(i, loss)
        print "Sample text:"
        x = X[0]
        sample_text = lstm.sample(1000, x, state, h)
        print "".join(map(lambda r: d_o[r], sample_text))
        print "\n"

# Close the document being trained on
doc_scanner.close_file()

