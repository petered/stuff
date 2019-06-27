import torch
from torch.optim import Adam

from petes_nns.pytorch_vae.pytorch_vaes import VAEModel, make_mlp_encoder, make_mlp_decoder


def test_pytorch_vae():

    n_samples = 10
    n_vis = 5
    n_latent = 3

    x = (torch.randn(n_samples, n_vis)>0).float()

    model = VAEModel(
        encoder = make_mlp_encoder(visible_dim=n_vis, hidden_sizes=[20, 20], latent_dim=n_latent),
        decoder = make_mlp_decoder(latent_dim=n_latent, hidden_sizes=[20, 20], visible_dim=n_vis, dist_type='bernoulli'),
        latent_dim=n_latent,
    )
    optimizer = Adam(params = model.parameters())

    for t in range(1000):
        optimizer.zero_grad()
        loss = -model.elbo(x).mean()

        print(loss.item())
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    test_pytorch_vae()

