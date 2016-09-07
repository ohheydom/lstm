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
        self.ih = self.init_weights([hidden_size, hidden_size])
        self.ib = self.init_biases([1, hidden_size])
        self.fx = self.init_weights([vocab_size, hidden_size])
        self.fh = self.init_weights([hidden_size, hidden_size])
        self.fb = self.init_biases([1, hidden_size])
        self.ox = self.init_weights([vocab_size, hidden_size])
        self.oh = self.init_weights([hidden_size, hidden_size])
        self.ob = self.init_biases([1, hidden_size])
        self.ax = self.init_weights([vocab_size, hidden_size])
        self.ah = self.init_weights([hidden_size, hidden_size])
        self.ab = self.init_biases([1, hidden_size])

        self.W = self.init_weights([hidden_size, vocab_size])
        self.b = self.init_biases([1, vocab_size])
    
    def lstm_cell(self, X, h, state):
        """lstm_cell feed forwards once through the network and returns a 
        new output and updated state.

        Parameters
        ----------
        X : array
            vocab_size X hidden_size. Input
        h : array
            hidden_size X hidden_size. Previous output
        state : array
            1 X hidden_size

        Returns
        -------
        output : array
            The output
        state : array
            The updated state
        """
        input_gate = self.sigmoid(np.dot(X, self.ix) + np.dot(h, self.ih) + self.ib)
        forget_gate = self.sigmoid(np.dot(X, self.fx) + np.dot(h, self.fh) + self.fb)
        output_gate = self.sigmoid(np.dot(X, self.ox) + np.dot(h, self.oh) + self.ob)

        update = np.tanh(np.dot(X, self.ax) + np.dot(h, self.ah) + self.ab)
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
        h : array
            1 X hidden_size. The returned output after the final input
        a_p : dict
            Updated Adam Optimization parameters
        """
        xs, y, c, p, ig, fg, og, a, h = {}, {}, {}, {}, {}, {}, {}, {}, {}
        c[-1] = np.copy(hprev)
        h[-1] = np.copy(o)
        loss = 0

        # Forward Pass
        # The following two lines increase efficiency by allowing two large matrix multiplications rather than 8 separate ones
        x_weight_matrix = [self.ix, self.fx, self.ox, self.ax]
        h_weight_matrix = [self.ih, self.fh, self.oh, self.ah]

        # ig, fg, og = input, forget, and output gates
        # a = memory cell
        # c = state
        # h = output
        # y = pre-softmaxed target labels
        # p = softmaxed target labels

        for t in range(len(X)):
            xs[t] = X[t]
            iz, fz, oz, az = np.reshape(np.dot(xs[t], x_weight_matrix) + np.dot(h[t-1], h_weight_matrix), [4, 1, self.hidden_size])
            ig[t] = self.sigmoid(iz + self.ib)
            fg[t] = self.sigmoid(fz + self.fb)
            og[t] = self.sigmoid(oz + self.ob)
            a[t] = np.tanh(az + self.ab)
            c[t] = c[t-1]*fg[t] + ig[t]*a[t]
            h[t] = og[t]*np.tanh(c[t])
            y[t] = np.dot(h[t], self.W) + self.b
            p[t] = self.softmax(y[t])
            loss += -np.log(p[t][0][np.argmax(Y[t])])

        # Backprop
        de_ix, de_ih, de_ib = np.zeros_like(self.ix), np.zeros_like(self.ih), np.zeros_like(self.ib)
        de_ox, de_oh, de_ob = np.zeros_like(self.ox), np.zeros_like(self.oh), np.zeros_like(self.ob)
        de_fx, de_fh, de_fb = np.zeros_like(self.fx), np.zeros_like(self.fh), np.zeros_like(self.fb)
        de_ax, de_ah, de_ab = np.zeros_like(self.ax), np.zeros_like(self.ah), np.zeros_like(self.ab)
        de_dW = np.zeros_like(self.W)
        de_db = np.zeros_like(self.b)
        de_ctm1 = np.zeros_like(c[0])

        for t in reversed(range(len(X))):
            de_dp = self.cross_entropy_loss(Y[t], p[t])
            de_dW += np.dot(h[t].T, de_dp)
            de_db += de_dp

            dy_dh = self.W
            de_dh = np.dot(de_dp, dy_dh.T)

            de_dc = de_dh*og[t]*self.tanh_prime(c[t]) + de_ctm1

            de_dog = de_dh*np.tanh(c[t])
            de_dogac = de_dog * self.sigmoid_prime(og[t])
            de_ox += np.dot(np.reshape(xs[t], [-1, 1]), de_dogac)
            de_oh += np.dot(np.reshape(h[t-1], [-1, 1]), de_dogac)
            de_ob += de_dogac

            de_fg = de_dc*c[t-1]
            de_ctm1 = de_dc*fg[t]
            de_ig = de_dc*a[t]
            de_a = de_dc*ig[t]

            de_dfgac = de_fg*self.sigmoid_prime(fg[t])
            de_fx += np.dot(np.reshape(xs[t], [-1, 1]), de_dfgac)
            de_fh += np.dot(np.reshape(h[t-1], [-1, 1]), de_dfgac)
            de_fb += de_dfgac

            de_digac = de_ig*self.sigmoid_prime(ig[t])
            de_ix += np.dot(np.reshape(xs[t], [-1, 1]),de_digac)
            de_ih += np.dot(np.reshape(h[t-1], [-1, 1]),de_digac)
            de_ib += de_digac

            de_daac = de_a*self.tanh_prime(a[t])
            de_ax += np.dot(np.reshape(xs[t], [-1, 1]), de_daac)
            de_ah += np.dot(np.reshape(h[t-1], [-1, 1]), de_daac)
            de_ab += de_daac

        # AdamUpdate
        mix, vix, tix = a_p['ix']
        mih, vih, tih = a_p['ih']
        mib, vib, tib = a_p['ib']
        mfx, vfx, tfx = a_p['fx']
        mfh, vfh, tfh = a_p['fh']
        mfb, vfb, tfb = a_p['fb']
        mox, vox, tox = a_p['ox']
        moh, voh, toh = a_p['oh']
        mob, vob, tob = a_p['ob']
        max_, vax, tax = a_p['ax']
        mah, vah, tah = a_p['ah']
        mab, vab, tab = a_p['ab']
        mw, vw, tw = a_p['w']
        mb, vb, tb = a_p['b']

        a_p['ix'] = self.adam_optimizer(de_ix, self.ix, m=mix, v=vix, t=tix)
        a_p['ih'] = self.adam_optimizer(de_ih, self.ih, m=mih, v=vih, t=tih)
        a_p['ib'] = self.adam_optimizer(de_ib, self.ib, m=mib, v=vib, t=tib)
        a_p['fx'] = self.adam_optimizer(de_fx, self.fx, m=mfx, v=vfx, t=tfx)
        a_p['fh'] = self.adam_optimizer(de_fh, self.fh, m=mfh, v=vfh, t=tfh)
        a_p['fb'] = self.adam_optimizer(de_fb, self.fb, m=mfb, v=vfb, t=tfb)
        a_p['ox'] = self.adam_optimizer(de_ox, self.ox, m=mox, v=vox, t=tox)
        a_p['oh'] = self.adam_optimizer(de_oh, self.oh, m=moh, v=voh, t=toh)
        a_p['ob'] = self.adam_optimizer(de_ob, self.ob, m=mob, v=vob, t=tob)
        a_p['ax'] = self.adam_optimizer(de_ax, self.ax, m=max_, v=vax, t=tax)
        a_p['ah'] = self.adam_optimizer(de_ah, self.ah, m=mah, v=vah, t=tah)
        a_p['ab'] = self.adam_optimizer(de_ab, self.ab, m=mab, v=vab, t=tab)
        a_p['w'] = self.adam_optimizer(de_dW, self.W, m=mw, v=vw, t=tw)
        a_p['b'] = self.adam_optimizer(de_db, self.b, m=mb, v=vb, t=tb)

        return loss, c[len(X)-1], h[len(X)-1], a_p

    def sample(self, sequence_size, x, state, h):
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
        h : array
            hidden_size X hidden_size

        Returns
        -------
        seq : array of indices
            1 X sequence_size
        """
        temp_p = np.copy(state)
        temp_h = np.copy(h)
        seq = []
        for _ in range(sequence_size):
            seq.append(np.argmax(x))
            temp_h, temp_p = self.lstm_cell(x, temp_h, temp_p)
            n_letter = np.dot(temp_h, self.W)
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
        adam_params = {'w': tups, 'b': tups, 'ix': tups, 'ih': tups, 'fx': tups, 'fh': tups, 'ox': tups, \
                'oh': tups, 'ax': tups, 'ah': tups, 'ib': tups, 'ab': tups, 'fb': tups, 'ob': tups}
        return adam_params
