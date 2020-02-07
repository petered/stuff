from functools import partial

import numpy as np
from artemis.experiments import experiment_function
from artemis.general.numpy_helpers import get_rng
from artemis.general.speedometer import Speedometer
from artemis.ml.tools.neuralnets import initialize_weight_matrix
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from artemis.plotting.matplotlib_backend import MovingPointPlot

from petes_nns.dynamical_networks.utils.signal_processing import get_named_nonlinearity, identity, BinarySigmaDelta

"""
Ok, here we will try using predictive coding to get the network to converge to the same values it started with.

Current status: We've verified that our previous demos were total BS (all hidden units were saturated).  And that just
using SD modulation, we obtain quite a different optimum.  
"""



class Network(object):

    def __init__(self, w_hh, w_hy, w_yh, b_h, hidden_act, output_act, decay, y_enc = None, h_enc = None, y_dec = None, h_dec = None):

        self.w_hh = w_hh
        self.w_hy = w_hy
        self.w_yh = w_yh
        self.b_h = b_h

        self.y_enc = y_enc if y_enc is not None else identity
        self.h_enc = h_enc if h_enc is not None else identity
        self.y_dec = y_dec if y_dec is not None else identity
        self.h_dec = h_dec if h_dec is not None else identity

        self.f_hid = get_named_nonlinearity(hidden_act)
        self.f_out = get_named_nonlinearity(output_act)
        self.decay = decay

    @classmethod
    def from_init(cls, n_hidden, n_out, b_h=0, rng=None, scale = .1, symmetric=False, **kwargs):
        rng = get_rng(rng)
        # w_hh = scale*initialize_weight_matrix(n_in=n_hidden, n_out=n_hidden, rng=rng)*.5
        # w_hy = scale*initialize_weight_matrix(n_in=n_hidden, n_out=n_out, rng=rng)*.1
        # w_yh = scale*initialize_weight_matrix(n_in=n_out, n_out=n_hidden, rng=rng)*.1
        w_hh = scale*initialize_weight_matrix(n_in=n_hidden, n_out=n_hidden, rng=rng)*scale
        if symmetric:
            w_hh= .5*(w_hh + w_hh.T)

        w_hy = scale*initialize_weight_matrix(n_in=n_hidden, n_out=n_out, rng=rng)*scale
        w_yh = w_hy.T if symmetric else scale*initialize_weight_matrix(n_in=n_out, n_out=n_hidden, rng=rng)*scale
        return Network(w_hh=w_hh, w_hy=w_hy, w_yh=w_yh, b_h=b_h, **kwargs)

    # @profile
    def update(self, h, y, return_signals = False):

        h_signal = self.h_enc(self.f_hid(h))
        y_signal = self.y_enc(self.f_out(y))

        # if self.h_enc is not identity:

        dbplot(h_signal[0, :], f'h-sig{id(self)}', draw_every=20)
        dbplot(y_signal[0, :], f'y-sig{id(self)}', draw_every=20)



        # print (f'H-signal min: {np.min(h)} max: {np.max(h)}')

        dh = -h * self.decay + self.h_dec(h_signal @ self.w_hh + y_signal @ self.w_yh) + self.b_h
        dy = -y * self.decay + self.y_dec(h_signal @ self.w_hy)
        h = h+ dh
        y = y+ dy
        if return_signals:
            return (h, y), (h_signal, y_signal)
        else:
            return h, y


@experiment_function
def demo_settling_dynamics(
        n_hidden=50,
        n_out=5,
        minibatch_size = 1,
        decay = 0.05,
        scale = .3,
        hidden_act = 'tanh',
        output_act = 'tanh',
        draw_every = 10,
        n_steps = 2500,
        symmetric=True,
        seed = 128):
    """
    Here we use Predictive Coding and compare_learning_curves the convergence of a predictive-coded network to one without.
    """

    generic_constructor = partial(Network.from_init, symmetric=symmetric, n_hidden=n_hidden, n_out=n_out, scale=scale, hidden_act=hidden_act, output_act=output_act, decay=decay, rng=seed)

    net = generic_constructor()
    qnet = generic_constructor(y_enc = BinarySigmaDelta(), h_enc = BinarySigmaDelta())

    rng = get_rng(seed)

    h = rng.randn(minibatch_size, n_hidden)
    y = rng.randn(minibatch_size, n_out)

    hq, yq = h.copy(), y.copy()

    sp = Speedometer()
    for t in range(n_steps):

        (h, y), (hsig, ysig) = net.update(h, y, return_signals=True)
        (hq, yq), (hqsig, yqsig) = qnet.update(hq, yq, return_signals=True)

        if t%100==0:
            print(f'Rate: {sp(t+1)} iter/s')

        with hold_dbplots(draw_every=draw_every):
            # dbplot(hsig[0], 'Hidden Units', title= 'Hidden Units (b={})'.format(net.b_h))
            # dbplot(hqsig[0], 'Quantized H-Signals')
            # dbplot(yqsig[0], 'Quantized Y-Signals')


            dbplot(h[0], 'Hidden Units', title= 'Hidden Units (b={})'.format(net.b_h))
            dbplot(hq[0], 'Quantized Hidden Units', title= 'Quantized Hidden Units (b={})'.format(net.b_h))

            dbplot(y[0], 'Y Units', title= 'Y Units (b={})'.format(net.b_h))
            dbplot(yq[0], 'Quantized Y Units')

            dbplot(np.mean(np.abs(h-hq)), 'h-error', grid=True, plot_type = lambda: MovingPointPlot(buffer_len=200))
            dbplot(np.mean(np.abs(y-yq)), 'y-error', grid=True, plot_type = lambda: MovingPointPlot(buffer_len=200))
            # dbplot(np.mean(np.abs(y-yq)), 'y-error', axis='h-error', grid=True)


        # if t==0:
        #     time.sleep(10)
        #     g.write(plt.gcf())
        # dbplot(h[5, :5], 'Hidden Subset')
        # dbplot(y[5, :5], 'Output Subset')


demo_settling_dynamics.add_variant('limit_cycle',
        # n_in=100,
        n_hidden=100,
        n_out=10,
        minibatch_size = 16,
        decay = 0.9,
        scale = .5,
        hidden_act = 'tanh',
        seed = 1235)


if __name__ == '__main__':
    demo_settling_dynamics()
    # profile.print_stats()
