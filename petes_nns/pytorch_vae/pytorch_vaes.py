from typing import NamedTuple

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Distribution, kl_divergence
from torch.optim import Optimizer, Adam, Adadelta, Adagrad, Adamax, SGD, RMSprop

from petes_nns.pytorch_vae.interfaces import IImageToPositionEncoder, IPositionToImageDecoder


class VAESignals(NamedTuple):

    z_distribution: Distribution
    x_distribution: Distribution
    z_samples: torch.Tensor
    elbo: torch.Tensor
    data_log_likelihood: torch.Tensor
    kl_div: torch.Tensor


class VAEModel(nn.Module):

    def __init__(self, encoder: IImageToPositionEncoder, decoder: IPositionToImageDecoder, latent_dim: int):
        super(VAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._latent_dim = latent_dim

    @property
    def latent_dim(self):
        return self._latent_dim

    def encode(self, x) -> Distribution:
        return self.encoder(x)

    def decode(self, z) -> Distribution:
        return self.decoder(z)

    def prior(self) -> Distribution:
        return Normal(loc=torch.zeros(self._latent_dim), scale=1)

    def recon(self, x):
        z_distribution = self.encoder(x)
        z = z_distribution.rsample()
        distro = self.decode(z).log_prob(x)
        return distro.rsample(sample_shape = x.size())

    def compute_all_signals(self, x, z_samples = None) -> VAESignals:
        z_distribution = self.encoder(x)
        kl_div = kl_divergence(z_distribution, self.prior()).sum(dim=1)
        z_samples = z_distribution.rsample() if z_samples is None else z_distribution.sample() if (isinstance(z_samples, str) and z_samples=='no_reparametrization') else z_samples
        x_distribution = self.decode(z_samples)
        data_log_likelihood = x_distribution.log_prob(x).flatten(1).sum(dim=1)
        elbo = data_log_likelihood - kl_div
        return VAESignals(z_distribution = z_distribution, z_samples = z_samples, x_distribution = x_distribution, elbo = elbo, data_log_likelihood=data_log_likelihood, kl_div = kl_div)

    def elbo(self, x):
        return self.compute_all_signals(x).elbo

    def sample(self, n_samples):
        z = self.prior().rsample(sample_shape=(n_samples, ))
        x_dist = self.decode(z)
        return x_dist.sample()


def get_supervised_vae_loss(signals: VAESignals, target: torch.Tensor, supervision_factor: float) -> torch.Tensor:
    """
    Suppose we have an idea of the desired latent representation of a VAE.  We can smoothely interpolate between the
    regular VAE loss, and a supervised loss where we explicitely train the latent representation.
    :param signals: The signals produced by calling VAEModel.compute_all_signals
    :param target: An (n_samples, target_dim) array
    :param supervision_factor: A number between 0 and 1, with 0 indicating no supervision (regular VAE loss) and 1
        indicating full supervision (latent representation explicitely trained)
    :return: The reconstruction
    """
    target_dim = target.size()[1]
    latent_dim = signals.z_samples.size()[1]
    assert target_dim<=latent_dim, "Target dim must be <= latent dim."
    prior_on_remaining_dims = Normal(loc=torch.zeros(latent_dim-target_dim), scale=1)
    # Push the latent dims to match the corresponding target dims.  Latent dims without target dims will be encouraged to approach the prior.
    supervised_loss = \
        - Normal(loc = signals.z_distribution.mean[:, :2], scale=signals.z_distribution.scale[:, :2]).log_prob(target).sum(dim=1) \
        + kl_divergence(Normal(loc = signals.z_distribution.mean[:, 2:], scale=signals.z_distribution.scale[:, 2:]), prior_on_remaining_dims).sum(dim=1)
    latent_loss = (1-supervision_factor)*signals.kl_div + supervision_factor*supervised_loss
    return -signals.data_log_likelihood + latent_loss


class VAESignalsSupervised(NamedTuple):

    z_distribution: Distribution
    x_distribution: Distribution
    z_samples: torch.Tensor
    elbo: torch.Tensor


class SupervisedVAE(VAEModel):

    def compute_all_signals_supervised(self):
        z_distribution = self.encoder(x)
        kl_div = kl_divergence(z_distribution, self.prior())
        z_samples = z_distribution.rsample()
        x_distribution = self.decode(z_samples)
        data_log_likelihood = x_distribution.log_prob(x)
        elbo = data_log_likelihood.flatten(1).sum(dim=1) - kl_div.sum(dim=1)
        return VAESignals(z_distribution = z_distribution, z_samples = z_samples, x_distribution = x_distribution, elbo = elbo)



def get_named_nonlinearity(name):
    return {'relu': nn.ReLU, 'tanh': nn.Tanh}[name]()


def make_mlp(in_size, hidden_sizes, out_size = None, nonlinearity = 'relu'):

    net = nn.Sequential()
    last_size = in_size
    for i, size in enumerate(hidden_sizes):
        net.add_module('L{}-lin'.format(i+1), nn.Linear(in_features=last_size, out_features=size))
        net.add_module('L{}-nonlin'.format(i+1),  get_named_nonlinearity(nonlinearity))
        last_size = size
    if out_size is not None:
        net.add_module('L{}-lin'.format(len(hidden_sizes)+1), nn.Linear(in_features=last_size, out_features=out_size))
    return net


class DistributionLayer(nn.Module):

    class Types:
        NORMAL = 'normal'
        BERNOULLI = 'bernoulli'
        NORMAL_UNITVAR = 'normal_unitvar'

    @classmethod
    def from_dense(cls, in_features, out_features):
        transform_constructor = lambda: nn.Linear(in_features, out_features)
        return cls(transform_constructor)

    @classmethod
    def from_conv(cls, in_features, out_features, kernel_size, **kwargs):
        transform_constructor = lambda: nn.Conv2d(in_features, out_features, kernel_size=kernel_size, **kwargs)
        return cls(transform_constructor)

    @staticmethod
    def get_class(name) -> 'DistributionLayer':
        return {
            DistributionLayer.Types.NORMAL: NormalDistributionLayer,
            DistributionLayer.Types.BERNOULLI: BernoulliDistributionLayer,
            DistributionLayer.Types.NORMAL_UNITVAR: NormalDistributionUnitVarianceLayer
        }[name]


class NormalDistributionLayer(DistributionLayer):

    def __init__(self, transform_constructor, var_mode='centered_softplus'):
        super(NormalDistributionLayer, self).__init__()
        self.mean_layer = transform_constructor()
        self.logscale_layer = transform_constructor()
        self.var_mode = var_mode

    def forward(self, x):
        mu = self.mean_layer(x)
        logsigma = self.logscale_layer(x)
        if self.var_mode == 'exp':
            scale = torch.exp(logsigma)
        elif self.var_mode == 'centered_softplus':
            scale = torch.nn.functional.softplus(logsigma + .5) + 1e-8
        else:
            raise NotImplementedError(self.var_mode)
        return Normal(loc=mu, scale=scale)


class NormalDistributionUnitVarianceLayer(DistributionLayer):

    def __init__(self, transform_constructor):
        super(NormalDistributionUnitVarianceLayer, self).__init__()
        self.mean_layer = transform_constructor()

    def forward(self, x):
        mu = self.mean_layer(x)
        return Normal(loc=mu, scale=1.)


class BernoulliDistributionLayer(nn.Module):

    def __init__(self, transform_constructor):
        super(BernoulliDistributionLayer, self).__init__()
        self.logit_layer = transform_constructor()

    def forward(self, x):
        logits = self.logit_layer(x)
        return Bernoulli(logits = logits)


class NormalDistributionConvLayer(nn.Module):

    def __init__(self, transform_constructor):
        super(NormalDistributionConvLayer, self).__init__()
        self.mean_layer = transform_constructor()
        self.logscale_layer = transform_constructor()

    def forward(self, x):
        mu = self.mean_layer(x)
        logsigma = self.logscale_layer(x)
        return Normal(loc=mu, scale=torch.exp(logsigma))


class BernoulliDistributionConvLayer(nn.Module):

    def __init__(self, in_shape, out_features):
        super(BernoulliDistributionConvLayer, self).__init__()
        self.logit_layer = nn.Conv2d(in_channels=out_features)

    def forward(self, x):
        logits = self.logit_layer(x)
        return Bernoulli(logits = logits)


def make_mlp_encoder(visible_dim, hidden_sizes, latent_dim, nonlinearity ='relu'):
    net = make_mlp(in_size=visible_dim, hidden_sizes=hidden_sizes, nonlinearity=nonlinearity)
    mid_size = visible_dim if len(hidden_sizes) == 0 else hidden_sizes[-1]
    top_layer = NormalDistributionLayer(mid_size, latent_dim)
    net.add_module('z_dist', top_layer)
    return net


def make_mlp_decoder(latent_dim, hidden_sizes, visible_dim, nonlinearity ='relu', dist_type ='bernoulli'):
    net = make_mlp(in_size=latent_dim, hidden_sizes=hidden_sizes, nonlinearity=nonlinearity)
    mid_size = latent_dim if len(hidden_sizes) == 0 else hidden_sizes[-1]
    final_layer = {'normal': NormalDistributionLayer, 'bernoulli': BernoulliDistributionLayer}[dist_type](mid_size, visible_dim)
    net.add_module('output', final_layer)
    return net


def get_named_optimizer(name, params, args) -> Optimizer:
    return {opt.__name__.lower(): opt for opt in [Adam, Adadelta, Adagrad, Adamax, SGD, RMSprop]}[name.lower()](params=params, **args)
