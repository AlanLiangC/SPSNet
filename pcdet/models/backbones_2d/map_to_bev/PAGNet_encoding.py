import torch
import torch.nn as nn
from surface_uncertainty.model import Generate_center
import numpy as np
# from surface_uncertainty.model_V3 import Generate_center
# from visdom import Visdom
# viz = Visdom(server='http://127.0.0.1', port=8097)


class PAGNet_encoding(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES

        self.generator = Generate_center(model_cfg.MODEL)
        self.generator.load_params_from_file_wo_logger(filename=model_cfg.CKPT, to_cpu=True)
        self.generator.cuda()
        self.generator.eval()

        for p_param in self.generator.parameters():
            p_param.require_grad = False

    def forward(self, batch_dict):

        batch_dict = self.generator.forward(batch_dict, training = False)
        
        if batch_dict.get('encoder_xyz', None) is not None:
            batch_dict.pop('encoder_xyz')
            batch_dict.pop('encoder_coords')
            batch_dict.pop('sa_ins_preds')

        ####################################Delete Points##########################################
        delete_number = 500
        delete_method = 'stability' # random or stability
        new_points = []
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        stds = batch_dict['stds']
        objects_mask = batch_dict['fake_labels'] > 0
        for batch_idx in range(batch_size):
            batch_mask = points[:,0] == batch_idx
            batch_points = points[batch_mask]
            batch_objects_mask = objects_mask[batch_mask]
            batch_object_number = torch.sum(batch_objects_mask)

            batch_bg_points = batch_points[~batch_objects_mask]
            batch_fg_points = batch_points[batch_objects_mask]
            if batch_object_number > delete_number:
                if delete_method == 'random':
                    sample_idx = torch.randperm(batch_object_number)[:(batch_object_number - delete_number)]
                elif delete_method == 'stability':
                    batch_stds = stds[batch_idx,...]
                    batch_object_stds = batch_stds[batch_objects_mask]
                    _,sample_idx = torch.topk(batch_object_stds, (batch_object_number - delete_number))
                else:
                    raise NotImplementedError
                
                batch_select_fg_points = batch_fg_points[sample_idx]
                batch_new_points = torch.cat([batch_bg_points, batch_select_fg_points])
            else:
                batch_bg_number = torch.sum(~batch_objects_mask)
                sample_idx = torch.randperm(batch_bg_number)[:(batch_bg_number + batch_object_number - delete_number)]
                batch_new_points = batch_bg_points[sample_idx]

            new_points.append(batch_new_points)
                
        batch_dict['points'] = torch.cat(new_points,dim=0)
        ####################################Delete Points##########################################


        return batch_dict




        