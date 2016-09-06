import numpy as np
import math

class LSTM:
    """LSTM uses sequences as inputs and returns output sequences. The LSTM
    architecture maintains a state throughout training, allowing the model
    to "remember" previous inputs.
    Instructions:
        Insantiate the model with LSTM(vocab_size, hidden_size).
        Initialize two zero valued matrices of size 1 X hidden_size, the state
    matrix and the output matrix.
        Initialize the adam optimization parameters with the method
    build_adam_params().
        Train the model with the bptt method, repeatedly updating the state 
    and output matrices and the adam_optimization dictionary.
        To sample text, use sample(sample_size, input_0_vector, state, output) 
        where the state and output matrices are taken from the training step.
            
    Parameters
    ----------
    vocab_size : int
        Size of the input and output vectors
    hidden_size : int
        Size of the hidden state
    """

    def __init__(self, vocab_size=100, hidden_size=100):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        #Learnable Params
        self.ix = self.init_weights([vocab_size, hidden_size])
        self.io = self.init_weights([hidden_size, hidden_size])
        self.ib = self.init_biases([1, hidden_size])
        self.fx = self.init_weights([vocab_size, hidden_size])
        self.fo = self.init_weights([hidden_size, hidden_size])
        self.fb = self.init_biases([1, hidden_size])
        self.ox = self.init_weights([vocab_size, hidden_size])
        self.oo = self.init_weights([hidden_size, hidden_size])
        self.ob = self.init_biases([1, hidden_size])
        self.cx = self.init_weights([vocab_size, hidden_size])
        self.co = self.init_weights([hidden_size, hidden_size])
        self.cb = self.init_biases([1, hidden_size])

        self.W = self.init_weights([hidden_size, vocab_size])
        self.b = self.init_biases([1, vocab_size])
    
    def lstm_cell(self, X, o, state):
        """lstm_cell feed forwards once through the network and returns a 
        new output and updated state.

        Parameters
        ----------
        X : array
            vocab_size X hidden_size
        o : array
            hidden_size X hidden_size
        state : array
            1 X hidden_size

        Returns
        -------
        output : array
            The output
        state : array
            The updated state
        """
        input_gate = self.sigmoid(np.dot(X, self.ix) + np.dot(o, self.io) + self.ib)
        forget_gate = self.sigmoid(np.dot(X, self.fx) + np.dot(o, self.fo) + self.fb)
        output_gate = self.sigmoid(np.dot(X, self.ox) + np.dot(o, self.oo) + self.ob)

        update = np.tanh(np.dot(X, self.cx) + np.dot(o, self.co) + self.cb)
        state = forget_gate*state + update*input_gate

        return output_gate*np.tanh(state), state

    def bptt(self, X, Y, hprev, o, a_p):
        """bptt trains the weights by feeding a sequence through
        the model and updating the weights via BackPropagation Through
        Time. The weights are optimized with the Adam Optimization method.

        Parameters
        ---------
        X : array
            sequence_size X vocab_size. A sequence of one hot vectors to
        be fed through the LSTM model
        Y : array
            sequence_size X vocab_size. A sequence of one hot target vectors
        hprev : array
            1 X hidden. The most recent state
        o : : array
            1 X hidden. The most recent output
        a_p : dict
            Adam Update Parameters

        Returns
        -------
        loss : float
            The current loss
        state : array
            1 X hidden_size. The returned state after the final input
        outs : array
            1 X hidden_size. The returned output after the final input
        a_p : dict
            Updated Adam Optimization parameters
        """
        xs, ys, hs, ps, ig, fg, og, c, outs = {}, {}, {}, {}, {}, {}, {}, {}, {} # c is the update cell
        hs[-1] = np.copy(hprev)
        outs[-1] = np.copy(o)
        loss = 0

        # Forward Pass
        # The following two lines increase efficiency by allowing two large matrix multiplications rather than 8 separate ones
        x_weight_matrix = [self.ix, self.fx, self.ox, self.cx]
        o_weight_matrix = [self.io, self.fo, self.oo, self.co]

        for t in range(len(X)):
            xs[t] = X[t]
            iz, fz, oz, cz = np.reshape(np.dot(xs[t], x_weight_matrix) + np.dot(outs[t-1], o_weight_matrix), [4, 1, self.hidden_size])
            ig[t] = self.sigmoid(iz + self.ib)
            fg[t] = self.sigmoid(fz + self.fb)
            og[t] = self.sigmoid(oz + self.ob)
            c[t] = np.tanh(cz + self.cb) # a in papers
            hs[t] = hs[t-1]*fg[t] + ig[t]*c[t] # c in papers
            outs[t] = og[t]*np.tanh(hs[t]) # ht in papers
            ys[t] = np.dot(outs[t], self.W) + self.b
            ps[t] = self.softmax(ys[t])
            loss += -np.log(ps[t][0][np.argmax(Y[t])])

        # Backprop
        d_ix, d_io, d_ib = np.zeros_like(self.ix), np.zeros_like(self.io), np.zeros_like(self.ib)
        d_ox, d_oo, d_ob = np.zeros_like(self.ox), np.zeros_like(self.oo), np.zeros_like(self.ob)
        d_fx, d_fo, d_fb = np.zeros_like(self.fx), np.zeros_like(self.fo), np.zeros_like(self.fb)
        d_cx, d_co, d_cb = np.zeros_like(self.cx), np.zeros_like(self.co), np.zeros_like(self.cb)
        d_W = np.zeros_like(self.W)
        d_b = np.zeros_like(self.b)
        d_hstm = np.zeros_like(hs[0]) #d_hstminus1

        for t in reversed(range(len(X))):
            d_y = self.cross_entropy_loss(Y[t], ps[t])
            d_W += np.dot(outs[t].T, d_y)
            d_b += d_y

            d_out = self.W
            d_e = np.dot(d_y, d_out.T)

            d_hst = d_e*og[t]*self.tanh_prime(hs[t]) + d_hstm

            d_ogf = d_e*np.tanh(hs[t])
            d_oraw = d_ogf * self.sigmoid_prime(og[t])
            d_ox += np.dot(np.reshape(xs[t], [-1, 1]), d_oraw)
            d_oo += np.dot(np.reshape(outs[t-1], [-1, 1]), d_oraw)
            d_ob += d_oraw

            d_fg = d_hst*hs[t-1]
            d_hstm = d_hst*fg[t]
            d_ig = d_hst*c[t]
            d_c = d_hst*ig[t]

            d_fraw = d_fg*self.sigmoid_prime(fg[t])
            d_fx += np.dot(np.reshape(xs[t], [-1, 1]), d_fraw)
            d_fo += np.dot(np.reshape(outs[t-1], [-1, 1]), d_fraw)
            d_fb += d_fraw

            d_iraw = d_ig*self.sigmoid_prime(ig[t])
            d_ix += np.dot(np.reshape(xs[t], [-1, 1]),d_iraw)
            d_io += np.dot(np.reshape(outs[t-1], [-1, 1]),d_iraw)
            d_ib += d_iraw

            d_craw = d_c*self.tanh_prime(c[t])
            d_cx += np.dot(np.reshape(xs[t], [-1, 1]), d_craw)
            d_co += np.dot(np.reshape(outs[t-1], [-1, 1]), d_craw)
            d_cb += d_craw

        # AdamUpdate
        mix, vix, tix = a_p['ix']
        mio, vio, tio = a_p['io']
        mib, vib, tib = a_p['ib']
        mfx, vfx, tfx = a_p['fx']
        mfo, vfo, tfo = a_p['fo']
        mfb, vfb, tfb = a_p['fb']
        mox, vox, tox = a_p['ox']
        moo, voo, too = a_p['oo']
        mob, vob, tob = a_p['ob']
        mcx, vcx, tcx = a_p['cx']
        mco, vco, tco = a_p['co']
        mcb, vcb, tcb = a_p['cb']
        mw, vw, tw = a_p['w']
        mb, vb, tb = a_p['b']

        a_p['ix'] = self.adam_optimizer(d_ix, self.ix, m=mix, v=vix, t=tix)
        a_p['io'] = self.adam_optimizer(d_io, self.io, m=mio, v=vio, t=tio)
        a_p['ib'] = self.adam_optimizer(d_ib, self.ib, m=mib, v=vib, t=tib)
        a_p['fx'] = self.adam_optimizer(d_fx, self.fx, m=mfx, v=vfx, t=tfx)
        a_p['fo'] = self.adam_optimizer(d_fo, self.fo, m=mfo, v=vfo, t=tfo)
        a_p['fb'] = self.adam_optimizer(d_fb, self.fb, m=mfb, v=vfb, t=tfb)
        a_p['ox'] = self.adam_optimizer(d_ox, self.ox, m=mox, v=vox, t=tox)
        a_p['oo'] = self.adam_optimizer(d_oo, self.oo, m=moo, v=voo, t=too)
        a_p['ob'] = self.adam_optimizer(d_ob, self.ob, m=mob, v=vob, t=tob)
        a_p['cx'] = self.adam_optimizer(d_cx, self.cx, m=mcx, v=vcx, t=tcx)
        a_p['co'] = self.adam_optimizer(d_co, self.co, m=mco, v=vco, t=tco)
        a_p['cb'] = self.adam_optimizer(d_cb, self.cb, m=mcb, v=vcb, t=tcb)
        a_p['w'] = self.adam_optimizer(d_W, self.W, m=mw, v=vw, t=tw)
        a_p['b'] = self.adam_optimizer(d_b, self.b, m=mb, v=vb, t=tb)

        return loss, hs[len(X)-1], outs[len(X)-1], a_p

    def sample(self, sequence_size, x, state, o):
        """sample returns an array of indices of size 1 X sequence_size
        that correspond to characters sampled from a corpus. It uses the
        updated weights and the current state and output matrices to feed
        forward through the network and sample values.

        Parameters
        ---------
        sequence_size : int
            Size of the sequence to return
        x : array
            1 X vocab_size
            One hot vector of the character to begin sampling from
        state : array
            1 X hidden_size
        o : array
            hidden_size X hidden_size

        Returns
        -------
        seq : array of indices
            1 X sequence_size
        """
        temp_p = np.copy(state)
        temp_o = np.copy(o)
        seq = []
        for _ in range(sequence_size):
            seq.append(np.argmax(x))
            temp_o, temp_p = self.lstm_cell(x, temp_o, temp_p)
            n_letter = np.dot(temp_o, self.W)
            p_letter = np.exp(n_letter) / np.sum(np.exp(n_letter))
            ix = np.random.choice(range(self.vocab_size), p=p_letter.ravel())
            x = np.zeros(self.vocab_size)
            x[ix] = 1
        return seq

    def init_weights(self, shape):
        """init_weights initializes an array of uniformly distributed
        values between 0.0 and 1.0 in the given shape.

        Parameters
        ---------
        shape : array
        An array of integers

        Returns
        -------
        An array of size shape
        """
        return np.random.uniform(size=shape)*0.001

    def init_biases(self, shape):
        """init_biases initializes an array of zeros in the given shape.

        Parameters
        ---------
        shape : array
        An array of integers

        Returns
        -------
        An array of size shape
        """
        return np.zeros(shape)

    def softmax(self, X):
        """softmax calculutes the softmaxed values of an input.

        Parameters
        ---------
        X : array

        Returns
        -------
        An array of softmaxed values

        """
        out = np.exp(X)
        sums = np.sum(out, axis=1)
        return np.divide(out, np.reshape(sums, [-1, 1]))

    def cross_entropy_loss(self, y, y_):
        """cross_entropy_loss calculates the derivative of the softmax function run through
        the cross entropy function.

        Parameters
        ---------

        Returns
        -------

        """
        return y_ - y

    def sigmoid(self, X):
        """sigmoid returns the result of an array computed through the sigmoid function.

        Parameters
        ---------
        X : array

        Returns
        -------
        An array of values run through the function
        """
        return 1/(1+math.e**(-X))

    def sigmoid_prime(self, sig):
        """sigmoid_prime returns the derivative of the sigmoid function.

        Parameters
        ---------
        sig : An array of sigmoid values

        Returns
        -------
        An array of gradients run through the function
        """
        return sig*(1-sig)

    def tanh_prime(self, v):
        """tanh_prime returns the derivative of the tanh function.

        Parameters
        ---------
        v : array

        Returns
        -------
        An array of gradients run through the function
        """
        return 1/(np.cosh(v)**2)

    def gradient_descent_optimizer(self, grads, which, lr=1e-2):
        """gradient_descent_optimizer ...

        Parameters
        ---------
        grads :
        which :
        lr : float
        """
        which -= lr*grads

    def adam_optimizer(self, grads, which, lr=1e-2, b1=0.9, b2=0.99, m=0, v=0, t=0, eps=1e-8):
        """adam_optimizer optimizes the given weight matrix via the
        Adam Optimization algorithm.

        Parameters
        ---------
        grads :
        which :
        lr : float
            Learning Rate
        b1 : float
            Beta 1, an exponential decay rate
        b2 : float
            Beta 2, an exponential decay rate
        m : array
            1st moment vector
        v : array
            2nd moment vector
        t : int
            Timestep
        eps : float
            Epsilon
        """
        t += 1
        new_m = b1*m + (1-b1)*grads
        new_v = b2*v + (1-b2)*grads**2
        mhat = new_m/(1-b1**t)
        vhat = new_v/(1-b2**t)
        which -= lr*mhat/(np.sqrt(vhat)+eps)
        return new_m, new_v, t

    def vectorize_sequence(self, sequence, d):
        """vectorize_sequence takes in a sequence and returns one-hot input
        and target arrays for the LSTM.

        Parameters
        ---------
        sequence : string
            A sequence of characters
        d : dict
            A mapping of characters to indices

        Returns
        -------
        X : array
            sequence_length X vocab_size
        Y : array
            sequence_length X vocab_size
        """

        X = []
        Y = []
        n = len(d)

        for i, c in enumerate(sequence[:-1]):
            cc = np.zeros(n)
            cc[d[c]] = 1
            X.append(cc)
            yy = np.zeros(n)
            yy[d[sequence[i+1]]] = 1
            Y.append(yy)
        return X, Y

    def build_adam_params(self):
        """build_adam_params returns zero valued adam parameters to be used
        and updated during the weight update.
        
        Returns
        -------
        A dict of 14 keys containing a tuple of (0, 0, 0)
        """
        tups = (0, 0, 0)
        adam_params = {'w': tups, 'b': tups, 'ix': tups, 'io': tups, 'fx': tups, 'fo': tups, 'ox': tups, \
                'oo': tups, 'cx': tups, 'co': tups, 'ib': tups, 'cb': tups, 'fb': tups, 'ob': tups}
        return adam_params
