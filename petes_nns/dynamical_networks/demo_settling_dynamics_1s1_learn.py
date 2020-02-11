from typing import Optional

from artemis.experiments import experiment_function
from artemis.general.numpy_helpers import get_rng
from artemis.general.speedometer import Speedometer
from artemis.ml.tools.neuralnets import initialize_weight_matrix
from artemis.plotting.db_plotting import dbplot, hold_dbplots
import numpy as np

from dynamical_networks.utils.signal_processing import get_named_nonlinearity_derivative
from petes_nns.dynamical_networks.utils.signal_processing import get_named_nonlinearity, identity
from attr import attrs, attrib

"""
Moved from demo_settling_dynamics_9 in old repo
"""


@attrs
class NetworkState:
    h = attrib()
    x = attrib()


@attrs
class Network(object):

    w_hh = attrib()
    w_hx = attrib()
    w_xh = attrib()
    b_h = attrib()
    fh = attrib(type=str)
    fx = attrib(type=str)
    decay = attrib()
    input_influence = attrib(default=1.)
    learning_rate = attrib(default=None, type=Optional[float])
    dfh = attrib(init=False, default=None)
    dfx = attrib(init=False, default=None)

    def __attrs_post_init__(self):
        self.dfh = get_named_nonlinearity_derivative(self.fh)
        self.dfx = get_named_nonlinearity_derivative(self.fx)
        self.fh = get_named_nonlinearity(self.fh)
        self.fx = get_named_nonlinearity(self.fx)


    def init_state(self, minibatch_size, rng = None):
        rng = get_rng(rng)
        n_hidden, n_out = self.w_hx.shape
        return NetworkState(
            h = rng.randn(minibatch_size, n_hidden),
            x = rng.randn(minibatch_size, n_out)
        )

    @classmethod
    def from_init(cls, n_hidden, n_out, b_h=0, rng=None, scale = .1, symmetric=False, **kwargs):
        rng = get_rng(rng)
        w_hh = scale*initialize_weight_matrix(n_in=n_hidden, n_out=n_hidden, rng=rng)
        if symmetric:
            w_hh= .5*(w_hh + w_hh.T)
        w_hx = scale*initialize_weight_matrix(n_in=n_hidden, n_out=n_out, rng=rng)
        w_xh = w_hx.T if symmetric else scale*initialize_weight_matrix(n_in=n_out, n_out=n_hidden, rng=rng)
        return Network(w_hh=w_hh, w_hx=w_hx, w_xh=w_xh, b_h=b_h, **kwargs)

    def update(self, state: NetworkState, inp=None) -> NetworkState:
        h_signal = self.fh(state.h)
        x_signal = self.fx(state.x)

        x_force = (inp-state.x)*self.input_influence if inp is not None else 0

        dh = -state.h * self.decay + h_signal @ self.w_hh + x_signal @ self.w_xh + self.b_h
        dx = -state.x * self.decay + h_signal @ self.w_hx + x_force

        if self.learning_rate is not None:
            self.w_hx += self.learning_rate * h_signal.T @ dx * self.dfx(state.x)
            # self.w_hh = self.learning_rate * h_signal.T @ dh
            # self.w_xh = self.learning_rate * x_signal.T @ dh
            # self.b_h = self.learning_rate * h_signal.mean(axis=0)

        return NetworkState(h =state.h + dh, x =state.x + dx)


@experiment_function
def demo_settling_dynamics(
        symmetric=False,
        n_hidden=50,
        n_out=3,
        input_influence = 0.01,
        learning_rate = 0.0001,
        cut_time = None,
        minibatch_size = 1,
        decay = 0.05,
        scale = .4,
        hidden_act = 'tanh',
        output_act = 'lin',
        draw_every = 10,
        n_steps = 10000,
        seed = 124
    ):
    """
    Here we use Predictive Coding and compare_learning_curves the convergence of a predictive-coded network to one without.
    """

    rng = get_rng(seed)
    net_d = Network.from_init(symmetric=symmetric, n_hidden=n_hidden, n_out=n_out, scale=scale,
                            fh=hidden_act, fx=output_act, decay=decay, rng=rng)
    state_d = net_d.init_state(minibatch_size=minibatch_size)

    net_l = Network.from_init(symmetric=symmetric, n_hidden=n_hidden, n_out=n_out, scale=scale,
                              fh=hidden_act, fx=output_act, decay=decay, rng=rng,
                              input_influence=input_influence, learning_rate = learning_rate)
    state_l = net_l.init_state(minibatch_size=minibatch_size)

    sp = Speedometer()
    for t in range(n_steps):

        error = (state_d.x[0]-state_l.x[0]).mean()
        with hold_dbplots(draw_every=draw_every):
            dbplot(state_d.h[0], 'hd')
            dbplot(state_d.x[0], 'xd')
            dbplot(state_l.h[0], 'hl')
            dbplot(state_l.x[0], 'xl')
            dbplot(np.array([abs(net_l.w_hx).mean()]), 'wmag')
            dbplot(error, 'error')



        state_d = net_d.update(state_d)
        state_l = net_l.update(state_l, inp=state_d.x if cut_time is None or t < cut_time else None)

        if t%100==0:
            print(f'Rate: {sp(t+1)} iter/s')



demo_settling_dynamics.add_variant('chaos', scale=0.5)
demo_settling_dynamics.add_variant('limit_cycle', scale=0.2)
demo_settling_dynamics.add_variant('fixed_point', symmetric=True)



if __name__ == '__main__':

    # demo_settling_dynamics()
    # demo_settling_dynamics.get_variant('chaos').call()
    demo_settling_dynamics(scale=0.5, cut_time=2000)
    # demo_settling_dynamics.get_variant('limit_cycle').call()
    # demo_settling_dynamics.get_variant('fixed_point').call()
