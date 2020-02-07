import itertools
from abc import abstractmethod

import numpy as np

from artemis.general.numpy_helpers import get_rng
from artemis.ml.parameter_schedule import ParameterSchedule
from petes_nns.dynamical_networks.utils.signal_processing import BinaryThresholder, identity, ExponentialMovingAverage


class PIDDecoder(object):

    def __init__(self, kp, ki=0., kd=0.):
        self.kd = kd
        self.ki = ki
        self.one_over_kpid = 1./float(kp + ki + kd) if (kp + ki + kd)>0 else np.inf
        # self.kd_minus_ki = kd-ki
        self.xp = 0.
        self.sp = 0.

    def __call__(self, y):
        x = self.one_over_kpid * (y - self.ki*self.sp + self.kd*self.xp)
        self.sp += x
        self.xp = x
        return x


class PIDEncoder(object):

    def __init__(self, kp, ki=0., kd=0., noise = 0., rng = None):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.xp = 0
        self.s = 0
        self.noise = noise
        self.rng = get_rng(rng)

    def __call__(self, x):
        self.s += x
        y = self.kp*x + self.ki*self.s + self.kd*(x-self.xp)
        self.xp = x.copy()
        if self.noise!=0:
            y += self.rng.randn(*y.shape)*self.noise
        return y


class MemoryScaleEncoder(object):

    def __init__(self, memory, scale):
        # assert 0<=memory<=1
        self.memory_gen = (memory for _ in itertools.count(0)) if isinstance(memory, (int, float)) else memory
        self.scale_gen = (scale for _ in itertools.count(0)) if isinstance(scale, (int, float)) else scale
        self.x_last = 0

    def __call__(self, x):
        z = next(self.scale_gen) * (x - next(self.memory_gen)*self.x_last)
        self.x_last = x
        return z


class MemoryScaleDecoder(object):

    def __init__(self, memory, scale):
        # assert 0<=memory<=1
        self.memory_gen = (memory for _ in itertools.count(0)) if isinstance(memory, (int, float)) else memory
        self.scale_gen = (scale for _ in itertools.count(0)) if isinstance(scale, (int, float)) else scale
        self.x_last = 0

    def __call__(self, q):
        self.x_last = q/next(self.scale_gen) + next(self.memory_gen)*self.x_last
        return self.x_last


class SaturatingDifferenceEncoder(object):

    def __init__(self, kd, delta_max = 1):
        self.kd = kd
        self.delta_max = delta_max
        self._x_last = 0

    def __call__(self, x):
        delta = x-self._x_last
        self._x_last = x
        return np.clip(x+self.kd*delta, 0, 1)


class PredictiveEncoder(object):

    def __init__(self, predictor_func, quantizer, scale=1):
        self.predictor_func = predictor_func
        self.quantizer = quantizer
        self.scale_func = scale if callable(scale) else (lambda q: scale)
        self._prediction = 0
        self._q = 0

    def __call__(self, x):
        scale = self.scale_func(self._q)
        err = scale*(x - self._prediction)
        q = self.quantizer(err)

        recon = q/scale + self._prediction
        self._prediction = self.predictor_func(recon)
        self._q = q
        return q


class PredictiveDecoder(object):

    def __init__(self, predictor_func, scale):
        self.predictor_func = predictor_func
        self.scale = scale
        self._prediction = 0

    def __call__(self, q):
        recon = q/self.scale + self._prediction
        self._prediction = self.predictor_func(recon)
        return recon


def get_predictive_encoder_decoder(predictor_constructor, scaler_constructor, quantizer):
    return PredictiveEncoder(predictor_func=predictor_constructor(), quantizer=quantizer, scale=scaler_constructor()), PredictiveDecoder(predictor_func=predictor_constructor(), scale=scaler_constructor())


def get_binary_encoder_decoder():

    class ScaleFunc(object):
        def __init__(self, s=1):
            self.s=s

        def __call__(self, q):
            s = self.s
            self.s = self.s*2
            return s

    return get_predictive_encoder_decoder(predictor_constructor=lambda: lambda x: x*2, scaler_constructor=ScaleFunc, quantizer=BinaryThresholder())



class MeanAndScaleDirectEncoder(object):

    def __init__(self, predictor, quantizer = None):
        """
        :param ScaleMemoryPredictor predictor:
        :param quantizer:
        """
        self.predictor = predictor
        self.quantizer = identity if quantizer is None else quantizer
        self.x_last = 0

    def __call__(self, x):
        mean, scale = self.predictor(self.x_last)
        z = (x-mean)*scale
        q = self.quantizer(z)
        self.x_last = x
        return q


class MeanAndScaleEncoder(object):

    def __init__(self, predictor, quantizer = None):
        """
        :param ScaleMemoryPredictor predictor:
        :param quantizer:
        """
        self.predictor = predictor
        self.quantizer = identity if quantizer is None else quantizer
        self.x_last = 0

    def __call__(self, x):
        mean, scale = self.predictor(self.x_last)
        z = (x-mean)*scale
        q = self.quantizer(z)
        self.x_last = mean + q/scale
        return q


class MeanAndScaleDecoder(object):

    def __init__(self, predictor):
        """
        :param ScaleMemoryPredictor predictor:
        """
        self.predictor = predictor
        self.x_last = 0.

    def __call__(self, z):
        mean, scale = self.predictor(self.x_last)
        x = z/scale + mean
        self.x_last = x
        return x


def get_mean_and_scale_predictive_encoder_decoder(predictor_constructor, quantizer, use_direct_encoder = False):
    """

    :param predictor_constructor:
    :param quantizer:
    :return:
    """
    if use_direct_encoder:
        encoder = MeanAndScaleDirectEncoder(predictor=predictor_constructor(), quantizer=quantizer)
    else:
        encoder= MeanAndScaleEncoder(predictor=predictor_constructor(), quantizer=quantizer)
    decoder = MeanAndScaleDecoder(predictor=predictor_constructor())
    return encoder, decoder


def get_binary_quantization_encoder_decoder():

    class Predictor(object):

        def __init__(self):
            self.mu = 0
            self.sigma = 1
            self.t = 0

        def __call__(self, x):
            q = int((x - self.mu)*2**self.t)>0.5

            self.sigma = self.sigma*2



class GrowingMemoryEncoder(object):

    def __init__(self, kp, kd_max, kd_tc):
        self.kp = kp
        self.kd_max = kd_max
        self.kd_tc = kd_tc
        self.x_last = 0
        self.t = 0

    def __call__(self, x):
        kd = (1-np.exp(-self.t/self.kd_tc)) * self.kd_max
        enc = self.kp * x + kd * (x - self.x_last)
        self.x_last = x
        self.t+=1
        return enc


class GrowingMemoryDecoder(object):

    def __init__(self, kp, kd_max, kd_tc):
        self.kp = kp
        self.kd_max = kd_max
        self.kd_tc = kd_tc
        self.x_last = 0
        self.t = 0

    def __call__(self, z):
        kd = (1-np.exp(-self.t/self.kd_tc)) * self.kd_max
        x = (z + kd * self.x_last)/(self.kp+kd)
        self.x_last = x
        self.t+=1
        return x


class BinaryEncoder(object):

    def __init__(self):
        self.t = 0
        self.last = 0

    def __call__(self, x):
        bit = (x - self.last) > 2**(-self.t-1)
        self.last = self.last + bit*2**-(self.t+1)
        self.t+=1
        return bit


class BinaryDecoder(object):

    def __init__(self):
        self.t = 1
        self.last = 0

    def __call__(self, bit):
        self.last = self.last + bit*(2**(-self.t))
        self.t+=1
        return self.last + (2**(-self.t))


class BinaryDecoder2(object):

    def __init__(self):
        self.t = 1
        self.avg = 0.5

    def __call__(self, q):
        self.avg = self.avg + 2**-self.t * (q-.5)
        self.t+=1
        return self.avg


class BinaryEncoder2(object):

    def __init__(self):
        self.dec = BinaryDecoder2()

    def __call__(self, x):
        q = x>self.dec.avg
        self.dec(q)
        return q




class AdaptiveMemoryScaleEncoder(object):
    # Unused now... delete?
    def __init__(self, max_scale, max_mem):
        self.max_scale = max_scale
        self.max_mem = max_mem
        self.x_last = 0

    def __call__(self, x):
        delta = np.exp(-(x - self.x_last))
        memory = (1-delta)*self.max_mem
        scale = (1/(1/self.max_scale + delta))
        z = scale * (x-memory*self.x_last)
        self.x_last = x
        return z


class AdaptiveMemoryScaleDecoder(object):
    # Unused now... delete?
    def __init__(self, max_scale, max_mem):
        self.max_scale = max_scale
        self.max_mem = max_mem
        self.x_last = 0

    def __call__(self, z):
        delta = np.exp(-(x - self.x_last))
        memory = (1-delta)*self.max_mem
        scale = (1/(1/self.max_scale + delta))
        z = scale * (x-memory*self.x_last)
        self.x_last = x
        return z


class ScaleMemoryPredictor(object):

    @abstractmethod
    def __call__(self, x):
        """
        :param x: The input to the predictor
        :return Tuple(Any, Any): The mean, scale
        """


class ConvergingScalePredictorOld(ScaleMemoryPredictor):

    def __init__(self, tc, final_scale, final_mem, initial_mem=0, initial_scale = 1):
        self.t = 0
        self.tc = tc
        self.final_scale = final_scale
        self.final_mem = final_mem
        self.initial_mem = initial_mem
        self.initial_scale = initial_scale

    def reset(self):
        self.t = 0

    def __call__(self, x):
        changiness = np.exp(-self.t/self.tc)  # Drops from 1 to zero as we converge
        memory = changiness*self.initial_mem + (1-changiness)*self.final_mem  # Exponentially converge

        scale = min(self.final_scale, self.initial_scale/changiness)

        # print(f'Memory: {memory:.3g}, Scale: {scale:.3g}')
        # scale = self.initial_scale * (1-changiness)
        # scale = min(self.final_scale, self.initial_scale/(changiness + (1.-changiness)/self.final_scale)
        mean = x * memory

        self.t+=1
        return mean, scale


class ConvergingScalePredictor(ScaleMemoryPredictor):

    AUTOSCALE = object()
    AUTOSCALE_MAX = object()

    def __init__(self, tc, scale_multiplier, data_scale):
        self.tc = tc
        self.scale_multiplier = scale_multiplier
        self.data_scale = \
            self.AUTOSCALE if isinstance(data_scale, str) and data_scale=='auto' else \
            self.AUTOSCALE_MAX if isinstance(data_scale, str) and data_scale=='automax' else \
            data_scale
        self.t = 0

    def reset(self):
        self.t = 0

    def __call__(self, x):
        scaled_exponent = self.scale_multiplier*np.exp(self.tc*self.t)

        data_scale = \
            x if self.data_scale is self.AUTOSCALE else \
            .005+np.max(x)*np.sign(np.argmax(x)) if self.data_scale is self.AUTOSCALE_MAX else \
            self.data_scale

        scale = scaled_exponent/data_scale
        # print(f'Scale: {scale}')
        memory = 1-1/(scaled_exponent)

        # dbplot(memory, f'{id(self)} memory', draw_now=False, plot_type=lambda: MovingPointPlot(buffer_len=60))
        # dbplot(scale, f'{id(self)} scale', draw_now=False, plot_type=lambda: MovingPointPlot(buffer_len=60))

        mean = x * memory
        self.t += 1
        return mean, scale

class DynamicAutoScalePredictorTransitional(ScaleMemoryPredictor):
    # TODO: revert once were done here

    def __init__(self, tc, scale_multiplier, data_scale):
        self.tc = tc
        self.scale_multiplier = scale_multiplier
        self.data_scale = data_scale
        self.t = 0
        self.x_last = 0

    def reset(self):
        self.t = 0

    def __call__(self, x):


        delta_ratio = np.abs(x-self.x_last).mean() / np.abs(x+self.x_last + 1e-9).mean()
        # dbplot(delta_ratio, f'{id(self)} delta r', draw_now=False, plot_type=lambda: MovingPointPlot(buffer_len=60))


        # OOOOKKK THIS ESTIMATOR WORKED OK WHEN DATA APPROPRIATELY SCALED.. DO NOT DELETE
        estimator = .04/(max(.002, np.abs(x-self.x_last).mean())) + self.scale_multiplier/2
        # dbplot(estimator, f'{id(self)} est', draw_now=False, plot_type=lambda: MovingPointPlot(buffer_len=60))

        self.x_last = x

        scaled_exponent = self.scale_multiplier*np.exp(self.tc*self.t)

        # dbplot(scaled_exponent, f'{id(self)} 1/sc', axis=f'{id(self)} est', draw_now=False, plot_type=lambda: MovingPointPlot(buffer_len=60, axes_update_mode='expand'), legend=['extimator', 'scaled exp'])

        # scaled_exponent = estimator

        scale = scaled_exponent/self.data_scale
        memory = 1-1./scaled_exponent
        mean = x * memory
        self.t += 1
        return mean, scale



class DynamicAutoScalePredictor(ScaleMemoryPredictor):
    # TODO: revert once were done here

    def __init__(self, scale_multiplier, data_scale):
        self.scale_multiplier = scale_multiplier
        self.data_scale = data_scale
        self.x_last = 0

    def __call__(self, x):

        estimator = 0.05/(np.abs(x-self.x_last).mean())
        if not np.isfinite(estimator):
            estimator = self.scale_multiplier

        self.x_last = x

        scaled_exponent = estimator
        scale = scaled_exponent/self.data_scale
        memory = 1-1/scaled_exponent
        mean = x * memory
        return mean, scale


# class ConvergingScalePredictor(object):
#
#     def __init__(self, tc,  max_scale, max_mem):
#         self.t = 0
#         self.tc = tc
#         self.max_scale = max_scale
#         self.max_mem = max_mem
#         self.x_last = 0
#         self.log_precision = 0
#
#     def __call__(self, x):
#         self.log_precision = self.t / self.tc
#         memory = (1-np.exp(-self.log_precision))*self.max_mem
#         scale = 1/(np.exp(-self.log_precision) + 1./self.max_scale)
#         mean = x * memory
#         self.t+=1
#         return mean, scale


class FixedRatePredictor(ScaleMemoryPredictor):

    def __init__(self, tc, max_scale, max_mem):
        self.t = 0
        self.tc = tc
        self.max_scale = max_scale
        self.max_mem = max_mem
        self.x_last = 0
        self.log_precision = 0

    def __call__(self, x):
        self.log_precision = self.t / self.tc
        memory = (1-np.exp(-self.log_precision))*self.max_mem
        scale = 1/(np.exp(-self.log_precision) + 1./self.max_scale)
        mean = x * memory
        self.t+=1
        return mean, scale




fcn = lambda x: ((x-.5)*2)**2 * 2 - 1


class ReconstructionBasedPredictiveEncoder(object):

    def __init__(self, mean, reconstructor):
        self.reconstructor = reconstructor
        self.mean = mean

    def __call__(self, x):
        q = x > self.mean
        self.mean = self.reconstructor(q)
        return q


class MeanIntervalReconstructor(object):

    def __init__(self,  mem, min_interval, max_interval=np.inf, mean=0.5, ):
        self.mem = mem
        self.mean = 0.5
        self.interval = 1.
        self.recent_bit_bias = mean
        self.min_interval = min_interval
        self.max_interval = max_interval

    def __call__(self, q):
        self.recent_bit_bias = self.mem * self.recent_bit_bias + (1 - self.mem) * q
        self.mean = self.mean + (q*2-1) * self.interval/4
        self.interval = np.minimum(self.max_interval, np.maximum(self.min_interval, self.interval * 2**fcn(self.recent_bit_bias)))
        return self.mean



class PeriodicMeanScalePredictor(ScaleMemoryPredictor):

    def __init__(self, scale_init, tc, period):
        self.scale_init = scale_init
        self.t = 0
        self.tc = tc
        self.period = period

    def reset(self):
        self.t = 0

    def __call__(self, z):
        scale = self.scale_init*np.exp((self.t%self.period)/self.tc)
        memory = 1 - 1/scale
        self.t +=1
        mean = memory*z
        return mean, scale


class MeanScaleScheduledPredictor(ScaleMemoryPredictor):

    def __init__(self, scale_schedule, mem_schedule):
        self.scale_schedule = ParameterSchedule(scale_schedule) if isinstance(scale_schedule, dict) else scale_schedule
        self.mem_schedule = ParameterSchedule(mem_schedule) if isinstance(mem_schedule, dict) else mem_schedule
        self.t = 0

    def reset(self):
        self.t = 0

    def __call__(self, x):
        mem = self.mem_schedule(self.t)
        scale = self.scale_schedule(self.t)
        mean = x*mem
        self.t = self.t+1
        return mean, scale


class LambdaScheduledPredictor(ScaleMemoryPredictor):

    def __init__(self, lambda_schedule, scale_scale=1):
        self.lambda_schedule = \
            lambda_schedule if callable(lambda_schedule) else \
            ParameterSchedule(lambda_schedule) if isinstance(lambda_schedule, dict) else \
            (lambda t: eval(lambda_schedule, dict(exp=np.exp, sqrt=np.sqrt), dict(t=t+1))) if isinstance(lambda_schedule, str) else \
            (lambda t: lambda_schedule)
        self.t = 0
        self.scale_scale = scale_scale

    def reset(self):
        self.t = 0

    def __call__(self, x):
        this_lambda = self.lambda_schedule(self.t)
        mem = (1-this_lambda)
        scale = 1/this_lambda
        mean = x*mem
        self.t = self.t+1
        return mean, scale*self.scale_scale


class HullMovingAverage(object):

    def __init__(self, decay):
        time_constant = 1./decay
        self.ema = ExponentialMovingAverage(decay=1/time_constant)
        self.ema2 = ExponentialMovingAverage(decay=1/(time_constant/2))
        self.emasqrt = ExponentialMovingAverage(decay=1/time_constant**.5)

    def __call__(self, x):
        return self.emasqrt(2*self.ema2(x) - self.ema(x))
