import torch
import torch.nn as nn
from typing import Any, List, Optional, Tuple, Union, Dict
import numpy as np

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

class AlanSparse2Dense():
    def __init__(self,
                 features: torch.Tensor,
                 indices: torch.Tensor,
                 spatial_shape: Union[List[int], np.ndarray],
                 batch_size: int,
                 grid: Optional[torch.Tensor] = None,
                 voxel_num: Optional[torch.Tensor] = None,
                 indice_dict: Optional[dict] = None,
                 benchmark: bool = False,
                 ):
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(v) for v in spatial_shape]
        self.batch_size = batch_size
        if indice_dict is None:
            indice_dict = {}
        self.indice_dict = indice_dict
        if grid is None:
            grid = torch.Tensor()  # empty tensor
        self.grid = grid
        self.voxel_num = voxel_num  # for tensorrt
        self.benchmark = benchmark
        self.benchmark_record = {}

    @property
    def features(self):
        return self._features

    def dense(self, channels_first: bool = True):
        output_shape = [self.batch_size] + list(
            self.spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(
            self.indices.to(self.features.device).long(), self.features,
            output_shape)
        if not channels_first:
            return res
        # ndim = len(self.spatial_shape)
        # trans_params = list(range(0, ndim + 1))
        # trans_params.insert(1, ndim + 1)
        return res.permute(0,3,1,2)


class Sparse2BEV(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.sparse_shape = grid_size[[1,0]]

    def forward(self, batch_dict, **kwargs):
        pillar_features = batch_dict['pillar_features']
        if batch_dict.get('pillar_coords', None) is not None:
            pillar_coords = batch_dict['pillar_coords']
        else:
            pillar_coords = batch_dict['voxel_coords'][:,[0,2,3]]
        batch_size = batch_dict['batch_size']
        
        input_sp_tensor = AlanSparse2Dense(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        ).dense()

        batch_dict['spatial_features'] = input_sp_tensor
        return batch_dict