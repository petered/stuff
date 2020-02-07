from abc import abstractmethod

import numpy as np
from artemis.general.numpy_helpers import get_rng
from artemis.ml.parameter_schedule import ParameterSchedule


class IQuantizer(object):

    @abstractmethod
    def __call__(self, x):
        """
        :param Array[float] x: Input signal
        :return Array[int]: Quantized output
        """
        pass


class SigmaDelta(IQuantizer):

    def __init__(self, coarseness=1.):
        self.phi = 0
        self.coarseness = coarseness

    def __call__(self, x):
        phi_ = self.phi + x*self.coarseness
        s = np.round(phi_)
        self.phi = phi_ - s
        return s/float(self.coarseness)


class BinarySampler(IQuantizer):

    def __init__(self, rng = None):
        self.rng = get_rng(rng)

    def __call__(self, x):
        return (x>self.rng.uniform(0, 1, size=x.shape)).astype(x.dtype)


class BinarySigmaDelta(IQuantizer):

    def __init__(self, precision=1., threshold = 0.5, values = None, phi_init=0, rng=None):
        assert values is None or len(values)==2
        self.rng = get_rng(rng)
        self.phi = phi_init
        self.precision = precision
        self.threshold = threshold
        self.values = values

    def __call__(self, x):
        if isinstance(self.phi, str) and self.phi == 'random':
            self.phi = self.threshold - self.rng.rand(*x.shape)
        phi_ = self.phi + x*self.precision
        s = (phi_>self.threshold).astype(x.dtype)
        if self.values is not None:
            s = np.where(s, self.values[1], self.values[0])
        self.phi = phi_ - s
        # self.phi = self.phi * 0.99

        return s/float(self.precision)


class NoisyBinarySigmaDelta(IQuantizer):

    def __init__(self, noise, precision=1., threshold = 0.5, values = None, phi_init=0, rng=None):
        assert values is None or len(values)==2
        self.rng = get_rng(rng)
        self.phi = phi_init
        self.precision = precision
        self.threshold = threshold
        self.values = values
        self.noise = noise

    def __call__(self, x):
        if isinstance(self.phi, str) and self.phi == 'random':
            self.phi = self.threshold - self.rng.rand(*x.shape)
        phi_ = self.phi + x*self.precision + self.noise*self.rng.randn(*x.shape)
        s = (phi_>self.threshold).astype(x.dtype)
        if self.values is not None:
            s = np.where(s, self.values[1], self.values[0])
        self.phi = phi_ - s
        # self.phi = self.phi * 0.99

        return s/float(self.precision)


class BinarySecondOrderSigmaDelta(IQuantizer):

    def __init__(self, threshold=0.5, phi_init=0):
        self.threshold = threshold
        self.phi_1 = phi_init
        self.phi_2 = phi_init

    def __call__(self, x):
        phi_1_ = self.phi_1 + x
        phi_2_ = self.phi_2 + phi_1_
        q = (phi_2_ > 0).astype(int)
        self.phi_2 = phi_2_ - q
        self.phi_1 = phi_1_ - q
        return q


class IdentityFunction(object):

    def __call__(self, x):
        return x


class BinarySigmaDeltaDynamicPrecision(IQuantizer):

    def __init__(self, precision_gen, threshold = 0.5, values = None):
        assert values is None or len(values)==2
        self.phi = 0
        self.precision_gen = precision_gen
        self.threshold = threshold
        self.values = values

    def __call__(self, x):
        precision = float(next(self.precision_gen))
        phi_ = self.phi + x*precision
        s = (phi_>self.threshold).astype(x.dtype)
        if self.values is not None:
            s = self.values[int(s)]
        self.phi = phi_ - s
        return s/precision




class BinaryThresholder(IQuantizer):

    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, x):
        return x>self.threshold


class BinaryStochasticRounder(IQuantizer):

    def __init__(self, seed=None):
        self.rng = get_rng(seed)

    def __call__(self, x):
        return np.array(x>self.rng.rand(*x.shape), copy=False).astype(int)

# class MemoryScaleEncoder(object):
#
#     def __init__(self, memory, scale):
#         self.memory = memory
#         self.scale = scale
#         self.x_last = 0
#
#     def __call__(self, x):
#         return self.scale * (x - self.memory * self.x_last)


class ExponentialMovingAverage(object):

    def __init__(self, decay, initial=0):
        self.avg = initial
        self.decay = decay

    def __call__(self, x):
        self.avg = (1-self.decay)*self.avg + self.decay*x
        return self.avg


class ScheduledExponentialMovingAverage(object):

    def __init__(self, decay_schedule, initial=0):
        self.decay_schedule = \
            decay_schedule if callable(decay_schedule) else \
            ParameterSchedule(decay_schedule) if isinstance(decay_schedule, dict) else \
            (lambda t: eval(decay_schedule, dict(exp=np.exp, sqrt=np.sqrt), dict(t=t))) if isinstance(decay_schedule, str) else \
            (lambda t: decay_schedule)
        self.avg = initial
        self.decay = decay_schedule
        self.t = 0

    def __call__(self, x):
        decay = self.decay_schedule(self.t)
        self.t+=1
        self.avg = (1-decay)*self.avg + decay*x
        return self.avg


class DoubleExponentialMovingAverage(object):

    def __init__(self, decay1, decay2=None, initial=0):
        self.avg1 = initial
        self.avg2 = initial
        self.decay1 = decay1
        self.decay2 = decay2 if decay2 is not None else decay1

    def __call__(self, x):
        self.avg1 = (1-self.decay1)*self.avg1 + self.decay1*x
        self.avg2 = (1-self.decay2)*self.avg2 + self.decay2*self.avg1
        return self.avg2


class ExponentialMovingSum(object):

    def __init__(self, decay, initial=0):
        self.avg = initial
        self.decay = decay

    def __call__(self, x):
        self.avg = (1-self.decay)*self.avg + x
        return self.avg.copy()


class Chain(object):

    def __init__(self, functions):
        self.functions = functions

    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x


class TemporalDifference(object):

    def __init__(self, initial=0):
        self.x_last = initial

    def __call__(self, x):
        delta = x - self.x_last
        self.x_last = x
        return delta



class OnlineDriftEstimate(object):

    def __init__(self, decay=0.1):
        self.ema = ExponentialMovingAverage(decay=decay)
        self.x_last = 0

    def __call__(self, x):
        estimate = self.ema(x - self.x_last)
        self.x_last = x
        return estimate


def identity_function(x):
    return x


def sigmoid(x):
    return 1./(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity(x):
    return x


def hardsig(x):
    return np.clip(x, 0, 1)

def get_named_nonlinearity(name):
    return {
        'sigm': sigmoid,
        'tanh': np.tanh,
        'relu': relu,
        'lin': identity,
        'hardsig': hardsig
    }[name]


def get_named_nonlinearity_derivative(name):
    return {
        'sigm': lambda x: sigmoid(x)*sigmoid(-x),
        'relu': lambda x: (x>=0).astype(float),
        'lin': lambda x: 1,
        'hardsig': lambda x: ((x>=0) & (x<=1)).astype(float),
        'tanh': lambda x: 1.0 - np.tanh(x)**2
    }[name]


class HoltWintersSmoothing(object):

    def __init__(self, alpha, beta):
        self.s = None
        self.b = 0
        self.alpha = alpha
        self.beta = beta

    def value_and_slope(self, x, alpha = None, beta=None):
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.bets
        if self.s is None:
            self.s = x
        s = alpha*x + (1-alpha)*(self.s + self.b)
        self.b = beta*(s-self.s) + (1-beta)*self.b
        self.s = s
        return s, self.b

    def __call__(self, x, alpha=None, beta=None):
        value, slope = self.value_and_slope(x, alpha=alpha, beta=beta)
        return value




class HoltWintersDriftAndNoiseEstimate(object):

    def __init__(self, alpha, beta):
        self.s = None
        self.b = 0
        self.alpha = alpha
        self.beta = beta
        self.var_est = 1.

    def __call__(self, x):
        if self.s is None:
            self.s = x

        self.var_est = (1-self.alpha)*self.var_est + self.alpha*(x - self.s - self.b)**2

        s = self.alpha*x + (1-self.alpha)*(self.s + self.b)
        self.b = self.beta*(s-self.s) + (1-self.beta)*self.b
        self.s = s
        return self.b, np.sqrt(self.var_est)