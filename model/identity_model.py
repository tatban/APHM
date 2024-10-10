import torch
import torch.nn as nn


def calc_gradient(y, x, grad_outputs=None):
    # for computing normals
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2,
                          (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2,
                          (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2,
                          (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2,
                          (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


class TripPlaneGenerator(nn.Module):
    def __init__(self, nz=512, ngf=64, nc=96, res=128, use_tanh=True):
        """
        Generates tri-planes from latent (shape) codes
        :param nz: input latent shape code dimension
        :param ngf: number of input channels for last transposed convolution before up sampling
        :param nc: three times number of channels in the generated (output) tri-planes. Default: 32x3=96
        :param res: spatial resolution of the generated (output) tri-planes [128(default) or 256]
        :param use_tanh: whether to use tanh at the last layer.
        """
        super(TripPlaneGenerator, self).__init__()
        self.use_tanh = use_tanh
        scale_factor = int(res / 64)
        assert scale_factor == 2 or scale_factor == 4, "tri-plane resolution must be 128 or 256"
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. bs x (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. bs x (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. bs x (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. bs x (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # state size. bs x (nc) x 64 x 64
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),  # state size. bs x (nc) x (res) x (res)
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),  # state size. bs x (nc) x res x res
        )

    def forward(self, z):
        if self.use_tanh:
            return torch.nn.functional.tanh(self.main(z))
        return self.main(z)



class MiniTriplaneLatent(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, channel_dim=32, res=256, latent_dim=512, cache_tri_planes=False, use_tanh=False, soft_plus_beta=100):
        super(MiniTriplaneLatent, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_size = channel_dim
        self.latent_dim = latent_dim
        self.lat_dim = latent_dim
        self.res = res
        if res == 128:
            self.tp_gen = TripPlaneGenerator(nz=self.latent_dim, nc=3*channel_dim, res=128, use_tanh=use_tanh)
        elif res == 256:
            self.tp_gen = TripPlaneGenerator(nz=self.latent_dim, nc=3 * channel_dim, res= 256, use_tanh=use_tanh)
        else:
            raise ValueError(f"Tri-Plane resolution has to be either 128 or 256. Got {res} instead.")
        self._n_shapes = None
        self._tri_planes = None
        self._previous_shape_codes = None
        self.cache_tri_planes = cache_tri_planes

        # tri-plane decoder
        self.net = nn.Sequential(
            nn.Linear(self.channel_size+self.input_dim, 256),  # with raw co-ordinates
            # nn.Linear(self.channel_size, 256),  # no raw co-ordinates
            nn.Softplus(beta=soft_plus_beta),
            nn.Linear(256, 256),
            nn.Softplus(beta=soft_plus_beta),
            nn.Linear(256, 256),
            nn.Softplus(beta=soft_plus_beta),
            nn.Linear(256, 256),
            nn.Softplus(beta=soft_plus_beta),
            nn.Linear(256, output_dim),
        )

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = grid_sample(
            plane,
            coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
        )
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H * W).permute(0, 2, 1)
        return sampled_features

    def generate_tri_planes(self, latent_code):
        tri_plane = self.tp_gen(latent_code)  # op dim: bsx96x128x128
        planes_list = torch.split(tri_plane, split_size_or_sections=self.channel_size, dim=1)
        assert planes_list[0].shape == planes_list[1].shape == planes_list[2].shape, "mal-formed tri-planes, check dimensions"
        self._tri_planes = planes_list
        return planes_list

    def reset_cache(self):  # IMPORTANT: needs to be called after optimizer.step in each iteration while using tri-plane caching
        if self.cache_tri_planes:
            self._tri_planes = None
            self._previous_shape_codes = None

    def _tp_Regenerate(self, current_code):  # tells when to generate tri-planes
        with torch.no_grad():
            return self._tri_planes is None or ((self._previous_shape_codes is not None) and (not torch.equal(current_code, self._previous_shape_codes)))

    def forward(self, coordinates, shape_code, debug=False):
        assert shape_code.ndim in [2, 4], "malformed_shape_code"
        if shape_code.ndim == 2:
            shape_code = shape_code.unsqueeze(2).unsqueeze(3)
        self._n_shapes, n_coords, n_dims = coordinates.shape  # bs x no of points x input_dim(=3)

        # todo: caching lc shape(identity) wise instead of batch wise, would be more useful for DIST rendering
        if self.cache_tri_planes:  # when tri-plane caching is enabled
            if self._tp_Regenerate(shape_code):  # cache miss
                planes_list = self.generate_tri_planes(shape_code)
                self._previous_shape_codes = shape_code.detach()
            else:  # cache hit
                planes_list = self._tri_planes
        else:  # when tri-plane caching is disabled
            planes_list = self.generate_tri_planes(shape_code)

        xy_embed = self.sample_plane(coordinates[..., 0:2], planes_list[0])  # op dim: bs x no of points x no of channels in tri-plane features
        yz_embed = self.sample_plane(coordinates[..., 1:3], planes_list[1])  # op dim: bs x no of points x no of channels in tri-plane features
        xz_embed = self.sample_plane(coordinates[..., :3:2], planes_list[2])  # op dim: bs x no of points x no of channels in tri-plane features

        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0,)  # op dim: bs x no of points x no of channels in tri-plane features
        inp = torch.cat([coordinates, features], dim=-1)
        return self.net(inp)  # op dim: bs x no of points x output_dim(=1 for SDF)
        # return self.net(features)  # op dim: bs x no of points x output_dim(=1 for SDF)

    def tvreg(self):
        l = 0
        for embed in self._tri_planes:
            l += ((embed[:, :, 1:] - embed[:, :, :-1]) ** 2).sum() ** 0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1]) ** 2).sum() ** 0.5
        return l

    def l2reg(self):
        l = 0
        for embed in self._tri_planes:
            l += (embed ** 2).sum() ** 0.5
        return l / self._n_shapes

    def edrreg(self, lc, num_points=10000, offset_distance=0.001):
        random_coords = torch.rand(lc.shape[0], num_points, 3).to("cuda") * 2 - 1  # sample from [-1, 1]
        offset_coords = random_coords + torch.randn_like(random_coords) * offset_distance  # Make offset_magnitude bigger if you want smoother
        densities_initial = self.forward(random_coords, lc)
        densities_offset = self.forward(offset_coords, lc)
        density_smoothness_loss = torch.nn.functional.mse_loss(densities_initial, densities_offset)
        return density_smoothness_loss

    def gradient(self, points, shape_code, sdf=None):
        if sdf is not None:
            grad = calc_gradient(sdf, points)
        else:
            points.requires_grad = True
            sdf = self.forward(points, shape_code)
            grad = calc_gradient(sdf, points)
        return grad

    def get_outputs(self, points, shape_code):
        points.requires_grad = True
        sdf = self.forward(points, shape_code)
        grad = calc_gradient(sdf, points)
        return sdf, grad

    def get_sdf_vals(self, points, shape_code):
        sdf = self.forward(points, shape_code)
        return sdf