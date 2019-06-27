import torch
from torch.optim import Adam, SGD, RMSprop, Adamax, Adagrad
from torchvision import datasets, transforms

from artemis.general.checkpoint_counter import Checkpoints
from artemis.general.ezprofile import EZProfiler
from artemis.general.global_rates import measure_global_rate, measure_rate_context, measure_runtime_context
from artemis.plotting.db_plotting import dbplot
from petes_nns.pytorch_vae.pytorch_vaes import VAEModel, make_mlp_encoder, make_mlp_decoder


def demo_pytorch_vae_mnist(
    hidden_sizes = [200, 200],
    latent_dim = 5,
    distribution_type = 'bernoulli',
    minibatch_size = 20,
    checkpoints = 100,
    n_epochs = 20
    ):

    cp = Checkpoints(checkpoints)

    model = VAEModel(
        encoder = make_mlp_encoder(visible_dim=784, hidden_sizes=hidden_sizes, latent_dim=latent_dim),
        decoder = make_mlp_decoder(latent_dim=latent_dim, hidden_sizes=hidden_sizes, visible_dim=784, dist_type=distribution_type),
        latent_dim=latent_dim,
    )
    # optimizer = Adam(params = model.parameters())
    # optimizer = RMSprop(params = model.parameters())
    # optimizer = Adamax(params = model.parameters())
    optimizer = Adagrad(params = model.parameters())
    # optimizer = SGD(lr=0.001, params = model.parameters())

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=minibatch_size, shuffle=True)

    for epoch in range(n_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):

            epoch_pt = epoch + batch_idx / len(train_loader)

            optimizer.zero_grad()
            loss = -model.elbo(x.flatten(1)).sum()
            loss.backward()
            optimizer.step()

            rate = measure_global_rate('training')

            if cp():

                print(f'Mean Rate at Epoch {epoch_pt:.2g}: {rate:.3g}iter/s')
                z_samples = model.prior().sample((64, ))
                x_dist = model.decode(z_samples)
                dbplot(x_dist.mean.reshape(-1, 28, 28), 'Sample Means', title=f'Sample Means at epoch {epoch_pt:.2g}')


if __name__ == '__main__':
    demo_pytorch_vae_mnist()