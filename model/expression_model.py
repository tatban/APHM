import torch
import torch.nn as nn
import numpy as np


class DeformationNetwork(nn.Module):  # design mostly adapted from DeepSDF model
    def __init__(
            self,
            lat_dim,
            hidden_dim,
            n_layers=8,
            geometric_init=False,
            radius_init=1,
            beta=100,
            out_dim=3,
            num_freq_bands=None,
            input_dim=3,
    ):
        """
        Given a point xyz and latent identity and expression code, this network predicts the delta
        change / deformation of the point from neutral canonical pose for that identity under
        that facial expression.
        :param lat_dim: identity code + expression code (concatenated) dimension
        :param hidden_dim: number of hidden neurons in each hidden layers in MLP
        :param n_layers: number of hidden layers
        :param geometric_init: whether to use geometric initialization for rough spherical.
        :param radius_init: radius of initial sphere for geometric init. No effect if geometric_init is False
        :param beta:beta parameter for soft-plus activation
        :param out_dim: output dimension
        :param num_freq_bands: number of frequency bands in fourier features. If None, Fourier features are not computed.
        :param input_dim: dimension of input points (xyz)
        """
        super(DeformationNetwork, self).__init__()
        if num_freq_bands is None:
            d_in_spatial = input_dim
        else:
            d_in_spatial = input_dim*(2*num_freq_bands+1)
        d_in = lat_dim + d_in_spatial
        self.lat_dim = lat_dim
        self.input_dim = input_dim
        print(d_in)
        print(hidden_dim)
        dims = [hidden_dim] * n_layers
        dims = [d_in] + dims + [out_dim]
        self.num_layers = len(dims)
        self.skip_in = [n_layers//2]
        self.num_freq_bands = num_freq_bands
        if num_freq_bands is not None:
            fun = lambda x: 2 ** x
            self.freq_bands = fun(torch.arange(num_freq_bands))
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)
            # if true preform geometric initialization (roughly spherical)
            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
            setattr(self, "lin" + str(layer), lin)
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, xyz, lat_rep):
        if self.num_freq_bands is not None:
            pos_embeds = [xyz]
            for freq in self.freq_bands:
                pos_embeds.append(torch.sin(xyz* freq))
                pos_embeds.append(torch.cos(xyz * freq))
            pos_embed = torch.cat(pos_embeds, dim=-1)
            inp = torch.cat([pos_embed, lat_rep], dim=-1)
        else:
            inp = torch.cat([xyz, lat_rep], dim=-1)
        x = inp
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.skip_in:
                x = torch.cat([x, inp], -1) / np.sqrt(2)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        return x