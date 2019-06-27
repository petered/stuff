import torch
from torch import nn
from torch.nn import functional as ff

from petes_nns.pytorch_vae.interfaces import IImageToPositionEncoder, IPositionToImageDecoder
from petes_nns.pytorch_vae.pytorch_helpers import get_default_device
from petes_nns.pytorch_vae.pytorch_vaes import DistributionLayer


def glorot_reinitialize_kernel_(param):

    # assert param.ndim()==4 # Conv kernel: (n_out, n_in, n_y, n_x)
    n_out, n_in, n_y, n_x = param.size()
    fan_in = n_in*n_y*n_x
    fan_out = n_out*n_y*n_x
    scale = torch.sqrt(torch.tensor(6./(fan_in +    fan_out)))
    param.data = scale * 2 * (torch.rand(param.size())-.5)


def reinit_conv2d_parameters(conv2dlayer: nn.Conv2d):
    w, b = conv2dlayer.parameters()
    glorot_reinitialize_kernel_(w)
    b.data[:] = 0


class ConvLSTMCell(nn.Module):

    def __init__(self, input_shape, output_channels, kernel_size=5, forget_bias=1.0):
        super(ConvLSTMCell, self).__init__()
        assert kernel_size % 2 ==1 , f'Kernel size must be odd.  Got: {kernel_size}'
        in_channels, self.size_y, self.size_x = input_shape
        self._out_channels = output_channels
        self.inp_conv = nn.Conv2d(in_channels = in_channels, out_channels=4*output_channels, kernel_size=kernel_size, padding = kernel_size//2)
        self.h_conv = nn.Conv2d(in_channels = output_channels, out_channels=4*output_channels, kernel_size=kernel_size, padding = kernel_size//2)

        reinit_conv2d_parameters(self.inp_conv)
        reinit_conv2d_parameters(self.h_conv)

        self._forget_bias = forget_bias

    def initial_state(self, n_samples):
        hidden_state = torch.zeros((n_samples, self._out_channels, self.size_y, self.size_x)).float().to(get_default_device())
        cell_state = torch.zeros((n_samples, self._out_channels, self.size_y, self.size_x)).float().to(get_default_device())
        return hidden_state, cell_state

    def forward(self, inp, state):
        hidden_state, cell_state = state
        midway_state = self.inp_conv(inp) + self.h_conv(hidden_state)
        input_gate, new_input, forget_gate, output_gate = torch.split(midway_state, midway_state.size()[1]//4, dim=1)
        new_cell_state = ff.sigmoid(forget_gate + self._forget_bias) * cell_state + ff.sigmoid(input_gate)*ff.tanh(new_input)
        new_hidden_state = ff.tanh(new_cell_state) * ff.sigmoid(output_gate)
        return new_hidden_state, (new_hidden_state, new_cell_state)


class ConvLSTMPositiontoImageDecoder(nn.Module, IPositionToImageDecoder):

    def __init__(self, input_shape, n_hidden_channels, n_canvas_channels, n_pose_channels=2, kernel_size=5, forget_bias=1.0, canvas_scale=4, n_steps = 12, output_kernel_size=5, output_type =DistributionLayer.Types.NORMAL_UNITVAR):
        super(ConvLSTMPositiontoImageDecoder, self).__init__()
        n_image_channels, self.n_y_canvas, self.n_x_canvas = input_shape
        self.canvas_scale = canvas_scale
        self.n_canvas_channels = n_canvas_channels
        self.lstm_cell = ConvLSTMCell(input_shape=(n_pose_channels, self.n_y_canvas // canvas_scale, self.n_x_canvas // canvas_scale), output_channels=n_hidden_channels, kernel_size=kernel_size, forget_bias=forget_bias)
        self.painter_conv = nn.ConvTranspose2d(n_hidden_channels, n_canvas_channels, kernel_size=self.canvas_scale, stride=self.canvas_scale)

        reinit_conv2d_parameters(self.painter_conv)

        self.n_steps = n_steps
        assert output_kernel_size%2==1, f'Output kernel size must be odd.  Got {output_kernel_size}'
        self.output_layer = DistributionLayer.get_class(output_type).from_conv(self.n_canvas_channels, n_image_channels, kernel_size=output_kernel_size, padding=output_kernel_size//2)

        reinit_conv2d_parameters(self.output_layer.mean_layer)

    def forward(self, poses):
        (h_state, c_state) = state = self.lstm_cell.initial_state(n_samples = len(poses))
        canvas = torch.zeros((len(poses), self.n_canvas_channels, self.n_y_canvas, self.n_x_canvas)).float().to(get_default_device())
        broadcast_poses = poses[:, :, None, None].repeat(1, 1, h_state.size()[2], h_state.size()[3])  # (n_samples, 2, size_y, size_x)
        for t in range(self.n_steps):
            out, state = self.lstm_cell(broadcast_poses, state)
            canvas = canvas + self.painter_conv(out)
        return self.output_layer(canvas)


class ConvLSTMImageToPositionEncoder(nn.Module, IImageToPositionEncoder):

    def __init__(self, input_shape, n_hidden_channels, n_pose_channels=2, kernel_size=5, forget_bias=1.0, canvas_scale=4, n_steps = 12, output_kernel_size=5, output_type = 'normal'):
        super(ConvLSTMImageToPositionEncoder, self).__init__()
        n_image_channels, self.n_y_canvas, self.n_x_canvas = input_shape
        self.canvas_scale = canvas_scale

        self.lstm_cell = ConvLSTMCell(input_shape=(n_hidden_channels, self.n_y_canvas//canvas_scale, self.n_x_canvas//canvas_scale), output_channels=n_hidden_channels, kernel_size=kernel_size, forget_bias=forget_bias)
        self.im_reader_conv = nn.Conv2d(n_image_channels, n_hidden_channels, kernel_size=self.canvas_scale, stride=self.canvas_scale)
        self.readout_conv = nn.Conv2d(n_hidden_channels, n_pose_channels, kernel_size=output_kernel_size)
        self.n_steps = n_steps
        self.mu_output_layer = nn.Conv2d(n_hidden_channels, n_pose_channels, kernel_size=output_kernel_size)
        self.logsig_output_layer = nn.Conv2d(n_hidden_channels, n_pose_channels, kernel_size=output_kernel_size)

    def forward(self, image):
        h_state, c_state = self.lstm_cell.initial_state(n_samples = len(image))
        img_remapped = self.im_reader_conv(image)
        for t in range(self.n_steps):
            h_state = h_state + img_remapped  # TODO: Why??? They seem to do this too though...
            out, (h_state, c_state) = self.lstm_cell(img_remapped, (h_state, c_state))

        mu = self.mu_output_layer(out).mean(dim=3).mean(dim=2)
        logsig = self.logsig_output_layer(out).mean(dim=3).mean(dim=2)
        return torch.distributions.Normal(loc=mu, scale=ff.softplus(logsig+.5) + 1e-8)
