# Long Short Term Memory Recurrent Neural Network

LSTM uses sequences as inputs and returns output sequences. The LSTM architecture maintains a state throughout training, allowing the model to "remember" previous inputs.

### Instructions

* Insantiate the model with `LSTM(vocab_size, hidden_size)`.
* Initialize two zero valued matrices of size 1 X hidden_size, the state matrix and the output matrix.
* Initialize the adam optimization parameters for training with the method `build_adam_params()`.
* Train the model with the `bptt` method`, repeatedly updating the state and output matrices and the adam_optimization dictionary.

### Sampling

To sample text, use `sample(sample_size, input_0_vector, state, output)` where the state and output matrices are taken from the training step.

### Examples

Please see `example.py` for an example on how to use the model.
