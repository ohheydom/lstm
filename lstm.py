import numpy as np
import math

class LSTM:
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
        """ Each gate will be of size:
            inputWeight = vocab, hidden_size
            outputWeight = hidden_size, hidden_size
            We add them together inside a sigmoid

            Both are initialized to 0:
            Output is size 1, hidden_size
            State is size 1, hidden_size
            
            Return output, state
        """
        input_gate = self.sigmoid(np.dot(X, self.ix) + np.dot(o, self.io) + self.ib)
        forget_gate = self.sigmoid(np.dot(X, self.fx) + np.dot(o, self.fo) + self.fb)
        output_gate = self.sigmoid(np.dot(X, self.ox) + np.dot(o, self.oo) + self.ob)

        update = np.tanh(np.dot(X, self.cx) + np.dot(o, self.co) + self.cb)
        state = forget_gate*state + update*input_gate

        return output_gate*np.tanh(state), state

    def loss_function(self, X, Y, hprev, o, a_p):
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
            c[t] = np.tanh(cz + self.cb) #a in papers
            hs[t] = hs[t-1]*fg[t] + ig[t]*c[t] #c in papers
            outs[t] = og[t]*np.tanh(hs[t]) #ht in papers
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
        a_p['ix'] = self.adam_optimizer(d_ix, self.ix, m=mix, v=vix, t=tix)

        mio, vio, tio = a_p['io']
        a_p['io'] = self.adam_optimizer(d_io, self.io, m=mio, v=vio, t=tio)

        mfx, vfx, tfx = a_p['fx']
        a_p['fx'] = self.adam_optimizer(d_fx, self.fx, m=mfx, v=vfx, t=tfx)

        mfo, vfo, tfo = a_p['fo']
        a_p['fo'] = self.adam_optimizer(d_fo, self.fo, m=mfo, v=vfo, t=tfo)

        mox, vox, tox = a_p['ox']
        a_p['ox'] = self.adam_optimizer(d_ox, self.ox, m=mox, v=vox, t=tox)

        moo, voo, too = a_p['oo']
        a_p['oo'] = self.adam_optimizer(d_oo, self.oo, m=moo, v=voo, t=too)

        mcx, vcx, tcx = a_p['cx']
        a_p['cx'] = self.adam_optimizer(d_cx, self.cx, m=mcx, v=vcx, t=tcx)

        mco, vco, tco = a_p['co']
        a_p['co'] = self.adam_optimizer(d_co, self.co, m=mco, v=vco, t=tco)

        mib, vib, tib = a_p['ib']
        a_p['ib'] = self.adam_optimizer(d_ib, self.ib, m=mib, v=vib, t=tib)

        mcb, vcb, tcb = a_p['cb']
        a_p['cb'] = self.adam_optimizer(d_cb, self.cb, m=mcb, v=vcb, t=tcb)

        mfb, vfb, tfb = a_p['fb']
        a_p['fb'] = self.adam_optimizer(d_fb, self.fb, m=mfb, v=vfb, t=tfb)

        mob, vob, tob = a_p['ob']
        a_p['ob'] = self.adam_optimizer(d_ob, self.ob, m=mob, v=vob, t=tob)

        mw, vw, tw = a_p['w']
        a_p['w'] = self.adam_optimizer(d_W, self.W, m=mw, v=vw, t=tw)

        mb, vb, tb = a_p['b']
        a_p['b'] = self.adam_optimizer(d_b, self.b, m=mb, v=vb, t=tb)

        #self.gradient_descent_optimizer(d_ix, self.ix)
        #self.gradient_descent_optimizer(d_io, self.io)
        #self.gradient_descent_optimizer(d_fx, self.fx)
        #self.gradient_descent_optimizer(d_fo, self.fo)
        #self.gradient_descent_optimizer(d_ox, self.ox)
        #self.gradient_descent_optimizer(d_oo, self.oo)
        #self.gradient_descent_optimizer(d_cx, self.cx)
        #self.gradient_descent_optimizer(d_co, self.co)
        #self.gradient_descent_optimizer(d_ib, self.ib)
        #self.gradient_descent_optimizer(d_cb, self.cb)
        #self.gradient_descent_optimizer(d_fb, self.fb)
        #self.gradient_descent_optimizer(d_ob, self.ob)
        #self.gradient_descent_optimizer(d_W, self.W)
        #self.gradient_descent_optimizer(d_b, self.b)

        return loss, hs[len(X)-1], outs[len(X)-1], a_p

    def sample(self, sequence, x, state, o):
        temp_p = np.copy(state)
        temp_o = np.copy(o)
        seq = []
        for _ in range(sequence):
            seq.append(np.argmax(x))
            temp_o, temp_p = self.lstm_cell(x, temp_o, temp_p)
            n_letter = np.dot(temp_o, self.W)
            p_letter = np.exp(n_letter) / np.sum(np.exp(n_letter))
            ix = np.random.choice(range(self.vocab_size), p=p_letter.ravel())
            x = np.zeros(self.vocab_size)
            x[ix] = 1
        return seq

    def init_weights(self, shape):
        return np.random.uniform(size=shape)*0.001

    def init_biases(self, shape):
        return np.zeros(shape)

    def softmax(self, X):
        out = np.exp(X)
        sums = np.sum(out, axis=1)
        return np.divide(out, np.reshape(sums, [-1, 1]))

    def cross_entropy_cost(self, y, y_):
        return np.mean(-y*np.log(y_))

    def cross_entropy_loss(self, y, y_):
        return y_ - y

    def sigmoid(self, X):
        return 1/(1+math.e**(-X))

    def sigmoid_prime(self, sig):
        return sig*(1-sig)

    def tanh_prime(self, v):
        return 1/(np.cosh(v)**2)

    def gradient_descent_optimizer(self, grads, which, lr=1e-2):
        which -= lr*grads

    def adam_optimizer(self, grads, which, lr=1e-2, b1=0.9, b2=0.99, m=0, v=0, t=0, eps=1e-8):
        t += 1
        new_m = b1*m + (1-b1)*grads
        new_v = b2*v + (1-b2)*grads**2
        mhat = new_m/(1-b1**t)
        vhat = new_v/(1-b2**t)
        which -= lr*mhat/(np.sqrt(vhat)+eps)
        return new_m, new_v, t

    def vectorize_sequence(self, sequence, d):
        """Returns two matrices
            Input: len(word) X vocab_size 
            Target: len(word) X vocab size
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
