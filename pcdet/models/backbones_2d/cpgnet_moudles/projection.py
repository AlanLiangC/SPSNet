#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    : 2022/06/24 09:31:52
@Author  : Hongda Chang
@Version : 0.1
'''
import math

import torch
import torch.nn as nn
from torch_scatter import scatter_max
# from torch_scatter import scatter_mean


def arctan2(x1, x2):
    x1_zero = x1 == 0
    x2_zero = x2 == 0
    x2_neg = x2 < 0

    # to prevent `nan` in output of `torch.arctan`.
    # we expect arctan(0/0) to be 0, and arctan(0/any) happens to be 0.
    x2 = x2 + (x1_zero & x2_zero)
    phi = torch.arctan(x1/x2)

    add_pi = ((x1 > 0) & x2_neg) | (x1_zero & x2_neg)
    neg_pi = (x1 < 0) & x2_neg

    phi += add_pi * math.pi
    phi -= neg_pi * math.pi

    return phi

class Projection:
    """P2G and G2P                 
        ----------->  u(x)
        |
        |
        |
        v  v(y)

    Param:
        pc_range: with shape :math:`(6, )`, that contain (x_min, y_min, z_min, x_max, y_max, z_max)
        pc_fov: rad of feild-of-view with shape :math:`(4, )`, that contain (v_down, v_up, h_left, h_right)
        bev_shape: with shape :math:`(2, )`, that contain (h, w)
        range_shape: with shape :math:`(2, )`, that contain (h, w)
    """

    def __init__(self, pc_range, pc_fov = None, bev_shape = None, range_shape = None):
        super(Projection, self).__init__()
        self.pc_range = pc_range
        self.pc_fov = pc_fov

        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        self.pc_bev_range = [x_min, y_min, x_max, y_max]
        if pc_fov is not None:
            v_down, v_up, h_left, h_right = pc_fov
            self.pc_vertical_fov = [v_down, v_up]
        
        self.bev_shape = bev_shape
        self.range_shape = range_shape

    def init_bev_coord(self, coord, eps=0.1):
        """Convert the coordinates to bev coordinates and save them as member variables
        
        Param:
            coord: points coordinates with shape :math:`(P, 4)` that contain (batch_index, x, y, z)
            eps: Prevent coordinate out of bounds of `self.bev_shape`

        Return:
            bev_coord: bev coordinate in valid range with shape :math:`(P_reduce, 3)`, that contain (batch_index, u, v)
            keep: mask of points in valid range with shape :math: `(P, )`
        """

        """
        # polar coordination projection
        h_bev, w_bev = [600, 600]
        radius_min, radius_max = 3, 50
        # x_min, y_min, x_max, y_max = self.pc_bev_range

        n, x, y, _ = coord.unbind(-1)
        depth = torch.norm(coord[:, 1:3], 2, 1)
        keep = (depth > radius_min) & (depth < radius_max)

        u = 0.5 * (1 - arctan2(y, x) / math.pi) *  w_bev
        v = (depth - radius_min) / (radius_max - radius_min) * h_bev
        u.clamp_(0, w_bev - eps)
        v.clamp_(0, h_bev - eps)

        self.bev_coord = torch.stack([n, u, v], dim=-1)[keep]
        return self.bev_coord, keep
        """

        h_bev, w_bev = self.bev_shape
        x_min, y_min, x_max, y_max = self.pc_bev_range

        n, x, y, _ = coord.unbind(-1)
        keep_x = torch.logical_and(x > x_min, x < x_max)
        keep_y = torch.logical_and(y > y_min, y < y_max)
        keep = torch.logical_and(keep_x, keep_y)

        u = (x - x_min) / (x_max - x_min) * w_bev
        v = (y - y_min) / (y_max - y_min) * h_bev
        u.clamp_(0, w_bev - eps) # make u to [0,w_bev - eps]
        v.clamp_(0, h_bev - eps)

        self.bev_coord = torch.stack([n, u, v], dim=-1)[keep]
        return self.bev_coord, keep

    def init_range_coord(self, coord, eps=0.1):
        """Convert the coordinates to range coordinates and save them as member variables
        
        Param:
            coord: points coordinates with shape :math:`(P, 4)` that contain (batch_index, x, y, z)
            eps: Prevent coordinate out of bounds of `self.range_shape`

        Return:
            range_coord: range coordinate in valid range with shape :math:`(P_reduce, 3)`, that contain (batch_index, u, v)
            keep: mask of points in valid range with shape :math: `(P, )`
        """
        h_range, w_range = self.range_shape
        v_down, v_up = self.pc_vertical_fov

        n, x, y, z = coord.unbind(-1)

        # r, theta, phi denote the distance, zenith and azimuth angle respectively
        r_sqr = x**2 + y**2 + z**2
        # r = torch.sqrt(r_sqr)
        theta = torch.arcsin(z / torch.sqrt(r_sqr + 1e-8))
        phi = arctan2(y, x)
        keep = torch.logical_and(theta > v_down, theta < v_up)

        u = 0.5 * (1 - phi / math.pi) * w_range
        v = (1 - (theta - v_down) / (v_up - v_down)) * h_range
        u.clamp_(0, w_range - eps)
        v.clamp_(0, h_range - eps)

        self.range_coord = torch.stack([n, u, v], dim=-1)[keep]
        return self.range_coord , keep

    def _scatter(self, points, grid_coord, batch_size, grid_shape):
        """
        Param:
            points: with shape :math:`(P_reduce, D)`, where P is points number, D is feature dimention.
            grid_coord: with shape :math:`(P_reduce, 3)`, that contain (batch_idx, u, v).
            batch_size: batch size.
            grid_shape: with shape :math:`(2, )`, that contain (h, w).
        
        Return:
            grid_map: with shape :math:`(N, C, H, W)` where N is batch size, H and W is shape of grid map. 
        """
        h_grid, w_grid = grid_shape
        n, u_floor, v_floor = grid_coord.long().unbind(-1)
        feature_dim = points.shape[-1]

        grid_coord_flatten = n * (h_grid * w_grid) + v_floor * w_grid + u_floor

        grid_map, _ = scatter_max(points, grid_coord_flatten.long(), 0, None, None)  # P_reduce * D 
        grid_map = torch.cat([grid_map, grid_map.new_zeros([int(batch_size * h_grid * w_grid - grid_map.shape[0]), feature_dim])])

        return grid_map.view(batch_size, int(h_grid), int(w_grid), feature_dim).permute(0, 3, 1, 2).contiguous()

    def _gather(self, grid_map, grid_coord, grid_shape):
        """
        Param:
            grid_map: with shape :math:`(N, C, H, W)` where N is batch size, H and W is shape of grid map.
            grid_coord: with shape :math:`(P_reduce, 3)`, that contain (batch_idx, u, v).
            grid_shape: with shape :math:`(2, )`, that contain (h, w).
        
        Return:
            points: with shape :math:`(P_reduce, D)`. 
        """
        h_grid, w_grid = grid_shape
        n, u, v = grid_coord.unbind(-1)
        n, u_floor, v_floor = n.long(), u.long(), v.long()
        u_floorp, v_floorp = u_floor + 1, v_floor + 1
        channels, h, w = grid_map.shape[1:]
        
        grid_map_ = grid_map.new_zeros([*grid_map.shape[:2], h+1, w+1])
        grid_map_[..., :h, :w] = grid_map
        grid_map = grid_map_
        h_grid, w_grid = h_grid + 1, w_grid + 1

        # (u, v) (u, v+) (u+ v) (u+, v+)
        bilinear_coord_flatten = n * (h_grid * w_grid) + torch.stack([v_floor, v_floorp, v_floor, v_floorp]) * w_grid + \
                                 torch.stack([u_floor, u_floor, u_floorp, u_floorp])  # 4 * N
        bilinear_weight_flatten = (1 - torch.abs(u - torch.stack([u_floor, u_floor, u_floorp, u_floorp]))) * \
                                  (1 - torch.abs(v - torch.stack([v_floor, v_floorp, v_floor, v_floorp])))  # 4 * N

        grid_map = grid_map.permute(0, 2, 3, 1).contiguous().view(-1, channels)[:, :, None].expand(-1, -1, 4)
        bilinear_coord_flatten = bilinear_coord_flatten.t()[:, None, :].expand(-1, channels, -1)
        bilinear_points = torch.gather(grid_map, 0, bilinear_coord_flatten)  # N, C, 4
        
        points = bilinear_weight_flatten.t().unsqueeze(1) * bilinear_points  # N, C, 4
        points = points.sum(dim=-1)  # N, C

        return points

    def p2g_bev(self, points, batch_size):
        """
        Param:
            points: with shape :math:`(P_reduce, D)`, where P_reduce is valid points number, D is feature dimention.
            batch_size: batch size. 
        
        Return:
            bev_map: with shape :math:`(N, C, H, W)` where N is batch size, H and W is bev feature map. 
        """
        bev_coord = getattr(self, 'bev_coord', None)
        assert bev_coord is not None, '`init_bev_coord` need to be first.'
        assert points.shape[0] == bev_coord.shape[0]

        return self._scatter(points, bev_coord, batch_size, self.bev_shape)

    def p2g_range(self, points, batch_size):
        """
        Param:
            points: with shape :math:`(P_reduce, D)`, where P_reduce is valid points number, D is feature dimention.
            batch_size: batch size. 
        
        Return:
            range_map: with shape :math:`(N, C, H, W)` where N is batch size, H and W is range feature map. 
        """
        range_coord = getattr(self, 'range_coord', None)
        assert range_coord is not None, '`init_range_coord` need to be first.'
        assert points.shape[0] == range_coord.shape[0]

        return self._scatter(points, range_coord, batch_size, self.range_shape)

    def g2p_bev(self, bev_map):
        """
        Param:
            bev_map:  with shape :math:`(N, C, H, W)` where N is batch size, C is number of channels, H and W is range feature map. 
        
        Return:
            points: with shape :math:`(P_ruduce, C)` where P_reduce is points number in specified range. 
        """
        bev_coord = getattr(self, 'bev_coord', None)
        assert bev_coord is not None, '`init_bev_coord` need to be first.'

        return self._gather(bev_map, bev_coord, self.bev_shape)

    def g2p_range(self, range_map):
        """
        Param:
            range_map:  with shape :math:`(N, C, H, W)` where N is batch size, C is number of channels, H and W is range feature map. 
        
        Return:
            points: with shape :math:`(P_ruduce, C)` where P_reduce is points number in specified range. 
        """
        range_coord = getattr(self, 'range_coord', None)
        assert range_coord is not None, '`init_range_coord` need to be first.'

        return self._gather(range_map, range_coord, self.range_shape)


    def preprocess(self, points: torch.Tensor, pc_range, bev_shape):
        """
        Param:
            points: with shape :math:`(P, 3+C)`, where P is points number, 3+C is (x, y, z) and other features.
            pc_range: list or tuple of (x_min, y_min, z_min, x_max, y_max, z_max).
            bev_shape: with shape :math:`(2, )`, that contain (h, w).
        
        Return:
            inputs_points: with shape :math:`(P, 3+C+2)`, where 3+C+2 is (x, y, z) and other features and delta x and delta y.
        """
        x, y, z = points[:, :3].unbind(-1)
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        h_bev, w_bev = bev_shape
        
        u = (x - x_min) / (x_max - x_min) * w_bev
        v = (y - y_min) / (y_max - y_min) * h_bev

        u_center = u.long() + 0.5
        v_center = v.long() + 0.5

        delta_x = u - u_center
        delta_y = v - v_center

        return torch.cat([points, delta_x, delta_y], dim=-1)
