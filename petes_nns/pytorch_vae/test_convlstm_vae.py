import torch
from torch.optim import Adam

from petes_nns.pytorch_vae.convlstm import ConvLSTMImageToPositionEncoder, ConvLSTMPositiontoImageDecoder
from petes_nns.pytorch_vae.pytorch_vaes import VAEModel


def test_convlstm_vae():

    n_samples, n_channels, size_y, size_x = 5, 3, 20, 20
    n_pose_channels = 2

    img = torch.randn(n_samples, n_channels, size_y, size_x)

    vae = VAEModel(
        encoder=ConvLSTMImageToPositionEncoder(input_shape=(n_channels, size_y, size_x), n_hidden_channels=20, n_pose_channels=n_pose_channels),
        decoder=ConvLSTMPositiontoImageDecoder(input_shape=(n_channels, size_y, size_x), n_hidden_channels=20, n_canvas_channels=20, n_pose_channels=n_pose_channels),
        latent_dim = n_pose_channels,
    )

    optimizer = Adam(params = vae.parameters())

    for t in range(1000):
        optimizer.zero_grad()
        loss = -vae.elbo(img).mean()

        print(loss.item())
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    test_convlstm_vae()
