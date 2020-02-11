from artemis.experiments import experiment_function
from artemis.general.numpy_helpers import get_rng
from artemis.general.speedometer import Speedometer
from artemis.ml.tools.neuralnets import initialize_weight_matrix
from artemis.plotting.db_plotting import dbplot, hold_dbplots

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
    fh = attrib(converter=get_named_nonlinearity)
    fx = attrib(converter=get_named_nonlinearity)
    decay = attrib()

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
        w_hy = scale*initialize_weight_matrix(n_in=n_hidden, n_out=n_out, rng=rng)
        w_yh = w_hy.T if symmetric else scale*initialize_weight_matrix(n_in=n_out, n_out=n_hidden, rng=rng)
        return Network(w_hh=w_hh, w_hx=w_hy, w_xh=w_yh, b_h=b_h, **kwargs)

    def update(self, state: NetworkState) -> NetworkState:
        h_signal = self.fh(state.h)
        x_signal = self.fx(state.x)
        dh = -state.h * self.decay + h_signal @ self.w_hh + x_signal @ self.w_xh + self.b_h
        dx = -state.x * self.decay + h_signal @ self.w_hx
        return NetworkState(h =state.h + dh, x =state.x + dx)


@experiment_function
def demo_settling_dynamics(
        symmetric=False,
        n_hidden=50,
        n_out=3,
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

    net = Network.from_init(symmetric=symmetric, n_hidden=n_hidden, n_out=n_out, scale=scale,
                            fh=hidden_act, fx=output_act, decay=decay, rng=seed)

    state = net.init_state(minibatch_size=minibatch_size)

    sp = Speedometer()
    for t in range(n_steps):
        state = net.update(state)

        if t%100==0:
            print(f'Rate: {sp(t+1)} iter/s')

        with hold_dbplots(draw_every=draw_every):

            dbplot(state.h[0], 'Hidden Units', title= 'Hidden Units (b={})'.format(net.b_h))
            dbplot(state.x[0], 'Y Units', title='Y Units (b={})'.format(net.b_h))


demo_settling_dynamics.add_variant('chaos', scale=0.5)
demo_settling_dynamics.add_variant('limit_cycle', scale=0.2)
demo_settling_dynamics.add_variant('fixed_point', symmetric=True)

if __name__ == '__main__':

    # demo_settling_dynamics()
    # demo_settling_dynamics.get_variant('chaos').call()
    demo_settling_dynamics(scale=0.1)
    # demo_settling_dynamics.get_variant('limit_cycle').call()
    # demo_settling_dynamics.get_variant('fixed_point').call()
