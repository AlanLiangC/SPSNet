import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
from torch.nn import Parameter, Softmax
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

# from point_net import PointNetfeat, SimPointNetfeat

from pcdet.utils import loss_utils
from pcdet.utils import common_utils
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_modules,surface_feature,pointnet2_utils
from pcdet.utils import box_coder_utils, box_utils, loss_utils, common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

class Surface_PW_feature(nn.Module):

    def __init__(self, model_cfg, input_channels=4, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        self.SF_extract = surface_feature.FeatureExtraction()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        sa_config = self.model_cfg
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        self.aggregation_mlps = sa_config.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = sa_config.get('CONFIDENCE_MLPS', None)
        self.max_translate_range = sa_config.get('MAX_TRANSLATE_RANGE', None)


        for k in range(sa_config.NSAMPLE_LIST.__len__()):
            if isinstance(self.layer_inputs[k], list): ###
                channel_in = channel_out_list[self.layer_inputs[k][-1]]
            else:
                channel_in = channel_out_list[self.layer_inputs[k]]

            mlps = sa_config.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            if self.aggregation_mlps and self.aggregation_mlps[k]:
                aggregation_mlp = self.aggregation_mlps[k].copy()
                if aggregation_mlp.__len__() == 0:
                    aggregation_mlp = None
                else:
                    channel_out = aggregation_mlp[-1]
            else:
                aggregation_mlp = None

            if self.confidence_mlps and self.confidence_mlps[k]:
                confidence_mlp = self.confidence_mlps[k].copy()
                if confidence_mlp.__len__() == 0:
                    confidence_mlp = None
            else:
                confidence_mlp = None

            self.SA_modules.append(
                pointnet2_modules.PointnetSampling(
                    npoint_list=sa_config.NPOINT_LIST[k],
                    sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                    sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                    radii=sa_config.RADIUS_LIST[k],
                    nsamples=sa_config.NSAMPLE_LIST[k],
                    mlps=mlps,
                    use_xyz=True,                                                
                    dilated_group=sa_config.DILATED_GROUP[k],
                    aggregation_mlp=aggregation_mlp)
            )

            channel_out_list.append(channel_out)


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None ###

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        li_cls_pred = None
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]

            ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
            li_xyz, li_features, sampled_idx_list = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz)

            encoder_xyz.append(li_xyz)
            # vs_points.append(li_xyz.view(batch_size, -1, 3)[1,...])
            li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            encoder_coords.append(torch.cat([li_batch_idx[..., None].float(),li_xyz.view(batch_size, -1, 3)],dim =-1))
            encoder_features.append(li_features)            
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1)) 
            else:
                sa_ins_preds.append([])

        surface_feature = self.SF_extract(xyz)
        
        # surface_feature = self.SF_extract(torch.cat([xyz, features.permute(0,2,1)], dim = -1))
        surface_feature = pointnet2_utils.gather_operation(surface_feature.permute(0,2,1).contiguous(), sampled_idx_list).permute(0,2,1).contiguous()
        pw_feature = encoder_features[-1].permute(0,2,1).contiguous()

        result = torch.cat([surface_feature, pw_feature], dim=-1)

        batch_dict['encoder_xyz'] = encoder_xyz
        batch_dict['encoder_coords'] = encoder_coords
        batch_dict['sa_ins_preds'] = sa_ins_preds
        # batch_dict['pw_feature'] = pw_feature
        # batch_dict['surface_feature'] = surface_feature
        batch_dict['soc_feature'] = result
        # batch_dict['sampled_idx_list'] = sampled_idx_list

        return batch_dict


# convert surface fearure to dist z
class Encoder_surface_feature(nn.Module):
    def __init__(self, input_channels, latent_size=3):
        super().__init__()

        self.fc1 = nn.Linear(input_channels, latent_size)
        self.fc2 = nn.Linear(input_channels, latent_size)

    def forward(self, features):
        mu = self.fc1(features)
        logvar = self.fc2(features)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)+3e-22), 1)

        return dist, mu, logvar
    

class Object_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, model_cfg):
        super(Object_feat_encoder, self).__init__()

        latent_dim = model_cfg.LATENT_DIM
        fe_out_channels = model_cfg.PW_FEATURE_DIM

        fc_scale = 0.25

        self.fc1 = nn.Linear(fe_out_channels + latent_dim, int(256 * fc_scale))
        self.fc2 = nn.Linear(int(256 * fc_scale), int(256 * fc_scale))

        self.fc_ce1 = nn.Linear(int(256 * fc_scale), int(256 * fc_scale))
        self.fc_ce2 = nn.Linear(int(256 * fc_scale), 3, bias=False)

    def forward(self, x, z):
        x = torch.cat([x, z], dim=-1)

        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))

        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        return centers

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)


class Generate_center(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.build_losses(self.model_cfg.LOSS_CONFIG)

        self.feature_extract = Surface_PW_feature(model_cfg.SA_CONFIG)
        self.feature_encoder = Encoder_surface_feature(input_channels=model_cfg.SF_FEATURE_DIM, latent_size=model_cfg.LATENT_DIM)
        self.obj_encoder = Object_feat_encoder(model_cfg.GENERATOR)

        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.last_batch_dict = None
        self.last_tb_dict = None

        if kwargs.get('training', None) is not None:
            self.training = kwargs['training']

    def update_global_step(self):
        self.global_step += 1


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def assign_stack_targets_IASSD(self, points, gt_boxes, extend_gt_boxes=None, weighted_labels=False,
                             ret_box_labels=False, ret_offset_labels=True,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0,
                             use_query_assign=False, central_radii=2.0, use_ex_gt_assign=False, fg_pc_ignore=False,
                             binary_label=False):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        box_idxs_labels = points.new_zeros(points.shape[0]).long() 
        gt_boxes_of_fg_points = []
        gt_box_of_points = gt_boxes.new_zeros((points.shape[0], 8))

        for k in range(batch_size):            
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            if use_query_assign: ##
                centers = gt_boxes[k:k + 1, :, 0:3]
                query_idxs_of_pts = roiaware_pool3d_utils.points_in_ball_query_gpu(
                    points_single.unsqueeze(dim=0), centers.contiguous(), central_radii
                    ).long().squeeze(dim=0) 
                query_fg_flag = (query_idxs_of_pts >= 0)
                if fg_pc_ignore:
                    fg_flag = query_fg_flag ^ box_fg_flag 
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = query_fg_flag
                    box_idxs_of_pts = query_idxs_of_pts
            elif use_ex_gt_assign: ##
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                extend_fg_flag = (extend_box_idxs_of_pts >= 0)
                
                extend_box_idxs_of_pts[box_fg_flag] = box_idxs_of_pts[box_fg_flag] #instance points should keep unchanged

                if fg_pc_ignore:
                    fg_flag = extend_fg_flag ^ box_fg_flag
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = extend_fg_flag 
                    box_idxs_of_pts = extend_box_idxs_of_pts 
                                
            elif set_ignore_flag: 
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            elif use_ball_constraint: 
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag

            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            bg_flag = (point_cls_labels_single == 0) # except ignore_id
            # box_bg_flag
            fg_flag = fg_flag ^ (fg_flag & bg_flag)
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]

            gt_boxes_of_fg_points.append(gt_box_of_fg_points)
            box_idxs_labels[bs_mask] = box_idxs_of_pts
            gt_box_of_points[bs_mask] = gt_boxes[k][box_idxs_of_pts]

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single


        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
            'box_idxs_labels': box_idxs_labels,
            'gt_box_of_points': gt_box_of_points,
        }
        return targets_dict
    
    def generate_center_ness_mask(self):
        pos_mask = self.forward_ret_dict['sa_ins_labels'][0] > 0
        gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points'][0]
        centers = self.forward_ret_dict['sa_xyz_coords'][0].view(-1,4)[:,1:]
        centers = centers[pos_mask].clone().detach()
        offset_xyz = centers[:, 0:3] - gt_boxes[:, 0:3]

        return pos_mask, offset_xyz

    def assign_targets(self, input_dict):
        target_cfg = self.model_cfg.TARGET_CONFIG
        gt_boxes = input_dict['gt_boxes']
        if gt_boxes.shape[-1] == 10:   #nscence
            gt_boxes = torch.cat((gt_boxes[..., 0:7], gt_boxes[..., -1:]), dim=-1)

        targets_dict_center = {}

        batch_size = input_dict['batch_size']      
        if target_cfg.get('EXTRA_WIDTH', False):  # multi class extension
            extend_gt = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=target_cfg.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])
        else:
            extend_gt = gt_boxes

        extend_gt_boxes = box_utils.enlarge_box3d(
            extend_gt.view(-1, extend_gt.shape[-1]), extra_width=target_cfg.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        if target_cfg.get('INS_AWARE_ASSIGN', False):
            sa_ins_labels, sa_gt_box_of_fg_points, sa_xyz_coords, sa_gt_box_of_points, sa_box_idxs_labels = [],[],[],[],[]
            sa_ins_preds = input_dict['encoder_coords']
            for i in range(1, len(sa_ins_preds)): 
                # if sa_ins_preds[i].__len__() == 0:
                #     continue
                sa_xyz = input_dict['encoder_coords'][i]
                if i == 1:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]  #[0.2, 0.2, 0.2]
                    ).view(batch_size, -1, gt_boxes.shape[-1])             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=True, use_ex_gt_assign= False 
                    )
                if i >= 2:
                # if False:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                    ).view(batch_size, -1, gt_boxes.shape[-1])             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=False, use_ex_gt_assign= True 
                    )

                sa_xyz_coords.append(sa_xyz)
                sa_ins_labels.append(sa_targets_dict['point_cls_labels'])
                sa_gt_box_of_fg_points.append(sa_targets_dict['gt_box_of_fg_points'])
                sa_gt_box_of_points.append(sa_targets_dict['gt_box_of_points'])
                sa_box_idxs_labels.append(sa_targets_dict['box_idxs_labels'])  

            targets_dict_center['sa_ins_labels'] = sa_ins_labels
            targets_dict_center['sa_gt_box_of_fg_points'] = sa_gt_box_of_fg_points
            targets_dict_center['sa_xyz_coords'] = sa_xyz_coords
            targets_dict_center['sa_gt_box_of_points'] = sa_gt_box_of_points
            targets_dict_center['sa_box_idxs_labels'] = sa_box_idxs_labels

        return targets_dict_center
    
    
    def build_losses(self, losses_cfg):

        if losses_cfg.LOSS_REG == 'WeightedSmoothL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
                    **losses_cfg.get('LOSS_REG_CONFIG', {})
                )
            )

        elif losses_cfg.LOSS_REG == 'WeightedL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
                )
            )
        else:
            raise NotImplementedError
        
    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div
        
    def get_training_loss(self):
        disp_dict = {}
        pos_mask, gt_ctr_offset = self.generate_center_ness_mask()
        pred_offset = self.forward_ret_dict['center_pred'][pos_mask]

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_weights = reg_weights[pos_mask]

        point_loss_box_src = self.reg_loss_func(
            pred_offset[None, ...], gt_ctr_offset[None, ...], weights=reg_weights[None, ...]
        )
        point_loss = point_loss_box_src.sum()
        tb_dict = {}

        tb_dict.update({'center_loss_box': point_loss.item()})

        regular_loss = l2_regularisation(self.feature_extract) + \
                l2_regularisation(self.feature_encoder) + l2_regularisation(self.obj_encoder)
        # ad hoc
        regular_loss = 5e-4 * regular_loss

        point_loss += regular_loss

        # dist loss
        mux = self.forward_ret_dict['mux'][pos_mask]
        logvarx = self.forward_ret_dict['logvarx'][pos_mask]

        normal_dist = Independent(Normal(loc=torch.zeros_like(mux), scale=torch.ones_like(logvarx)), 1)
        dist_prior = Independent(Normal(loc=mux, scale=torch.exp(logvarx)+3e-22), 1)

        lattent_loss = torch.mean(self.kl_divergence(normal_dist, dist_prior))*5e-2
        point_loss += lattent_loss

        mux = self.forward_ret_dict['mux'][~pos_mask]
        logvarx = self.forward_ret_dict['logvarx'][~pos_mask]

        normal_dist = Independent(Normal(loc=mux, scale=torch.ones_like(logvarx)*20), 1)
        dist_prior = Independent(Normal(loc=mux, scale=torch.exp(logvarx)+3e-22), 1)

        lattent_loss2 = torch.mean(self.kl_divergence(normal_dist, dist_prior))*5e-2
        point_loss += lattent_loss2


        disp_dict.update({
            'pos_mask': pos_mask,
            'lattent_loss': lattent_loss.item(),
            'lattent_loss2': lattent_loss2.item(),
        })

        return point_loss, tb_dict, disp_dict

    def save_vis_points(self, batch_dict, output_dir):

        self.forward_ret_dict = self.assign_targets(batch_dict)
        pos_mask, _ = self.generate_center_ness_mask()

        vs_points = []
        points = batch_dict['points'][:,:4]
        batch_mask = batch_dict['points'][:,0] == 0
        stds = batch_dict['stds']
        batch_points = points[batch_mask]

        vs_points.append(torch.cat([batch_points[:,1:4],batch_dict['fake_labels'][batch_mask].view(-1,1)], dim = -1))


        _, topk = torch.topk(-stds,4096,dim = 1)
        # stds = (stds/stds.max()).clamp(0,1)
        sa_points_wh_label = torch.cat([batch_dict['encoder_xyz'][-1], stds.unsqueeze(dim = -1)],dim=-1)
        
        vs_points.append(sa_points_wh_label[0,...])

        batch_mask = batch_dict['encoder_coords'][-1][:,:,0] == 0
        vs_points.append(sa_points_wh_label.view(-1,4)[pos_mask & batch_mask.view(-1)])

        vs_points.append(sa_points_wh_label[0,...][topk[0,...]])

        save_name = ['points','heat_map','instance_map','centain_points']

        file_name = batch_dict['frame_id']
        output_path = os.path.join(output_dir,file_name[0])
        os.makedirs(output_path, mode=0o777, exist_ok=True)

        for i in range(len(save_name)):
            np.savetxt(os.path.join(output_path,'{}.txt'.format(save_name[i])) , vs_points[i].detach().cpu().numpy())


    def forward(self, batch_dict, **kwargs):
        batch_size = batch_dict['batch_size']
        batch_dict = self.feature_extract(batch_dict)
        # surface_feature = batch_dict['surface_feature']
        # pw_feature = batch_dict['pw_feature']
        soc_feature = batch_dict['soc_feature']

        if kwargs.get('training', None) is not None:
            self.training = kwargs['training']

        if self.training:
            self.prior, mux, logvarx = self.feature_encoder(soc_feature)

            z_noise_prior = self.reparametrize(mux, logvarx)

            center_pred  = self.obj_encoder(soc_feature, z_noise_prior).view(-1,3)

            self.forward_ret_dict = self.assign_targets(batch_dict)

            self.forward_ret_dict.update({
                'center_pred': center_pred,
                'mux': mux.view(-1, mux.shape[-1]),
                'logvarx': logvarx.view(-1, logvarx.shape[-1])
            })

            loss, tb_dict, disp_dict = self.get_training_loss()

            return loss, tb_dict, disp_dict
        
        else:
            _, mux, logvarx = self.feature_encoder(soc_feature)

            stds = torch.sum(logvarx.mul(0.5).exp_(),dim = -1)
            batch_dict.update({
                'stds': stds
            })

            ######################################
            save_path = Path(os.path.join('.','SPSNet_vis_V3'))
            save_path.mkdir(parents=True, exist_ok=True)
            self.save_vis_points(batch_dict,save_path)
            ######################################

            return batch_dict
        
    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_from_file_wo_logger(self, filename, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

            
            





