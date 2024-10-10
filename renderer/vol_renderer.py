# Adapted from VolSDF: https://github.com/lioryariv/volsdf/tree/main/code/model

import torch
import torch.nn as nn
import numpy as np
from renderer.density import LaplaceDensity
from renderer.ray_sampler import ErrorBoundSampler
from renderer import rend_utils as rend_util


class Renderer(nn.Module):
    def __init__(self, implicit_network, conf, rend_res=64):
        super().__init__()
        self.scene_bounding_sphere = conf.get('scene_bounding_sphere', 1.0)
        self.white_bkgd = conf.get('white_bkgd', True)
        self.bg_color = torch.tensor(conf.get("bg_color", [1.0, 1.0, 1.0])).float().cuda()
        self.rend_res = rend_res
        uv = np.mgrid[0:self.rend_res, 0:rend_res].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().cuda()
        self.uv = uv.reshape(2, -1).transpose(1, 0).unsqueeze(0)
        self.implicit_network = implicit_network

        self.density = LaplaceDensity(**conf.get('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get('ray_sampler'))

    def forward(self, intrinsics, pose, shape_code):
        assert intrinsics.shape[0] == pose.shape[0], "batch dim missmatch in extrinsic and intrinsic"
        uv = self.uv.expand(pose.shape[0], -1, -1)  # dim: bs x res^2 x 2

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        ray_dirs *= -1.

        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None].expand(pose.shape[0], -1, -1), intrinsics)
        depth_scale = ray_dirs_tmp[:, :, 2:]

        batch_size, num_pixels, _ = ray_dirs.shape
        assert batch_size == uv.shape[0] and num_pixels == self.rend_res**2

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1)
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, shape_code)
        N_samples = z_vals.shape[-1]

        points = cam_loc.unsqueeze(2) + z_vals.unsqueeze(3) * ray_dirs.unsqueeze(2)
        points_flat = points.reshape(ray_dirs.shape[0], -1, 3)

        sdf, gradients = self.implicit_network.get_outputs(points_flat, shape_code)
        weights = self.volume_rendering(z_vals, sdf)
        depth_values = torch.sum(weights * z_vals, 2, keepdim=True) / (weights.sum(dim=2, keepdim=True) + 1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        output = {
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'grad_theta': gradients,
        }

        normals = gradients
        normals = normals.reshape(depth_values.shape[0], -1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 2)  # dim: bs x rend_res**2 x 3

        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            normal_map = normal_map + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)
        output['normal_map_flattened'] = normal_map
        # transform to local coordinate system
        output['normal_map'] = torch.flip(normal_map.permute(0, 2, 1).view(batch_size, 3, self.rend_res, self.rend_res), dims=(3,))

        return output, ray_dirs, cam_loc, points_flat

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(z_vals.shape)  # (batch_size * num_pixels) x N_samples
        dists = z_vals[:, :, 1:] - z_vals[:, :, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).unsqueeze(1).repeat(dists.shape[0], dists.shape[1], 1)], -1)
        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], dists.shape[1], 1).cuda(), free_energy[:, :, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance  # probability of the ray hits something here

        return weights
