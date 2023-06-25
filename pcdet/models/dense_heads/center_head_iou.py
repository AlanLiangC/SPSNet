import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils, box_utils, common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils



class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHeadIoU(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.rectifier = torch.tensor(model_cfg.POST_PROCESSING.get('RECTIFIER', 0.0)).view(-1).cuda()
        self.use_det_for_sem = model_cfg.get('USE_DET_FOR_SEM', False)
        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        if self.model_cfg.get('SEM_TASK', False):
            self.sem_criterion = loss_utils.CPGNetCriterion(
                    weight=self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['sem_cs_weight'], ignore=self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('sem_ignore',None),classes='present', with_ls=True, with_tc=False
                )

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())
        self.add_module('iou_loss_func', loss_utils.IouLoss())
        self.add_module('ins_loss_func', loss_utils.WeightedClassificationLoss())


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
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 or binary_label else gt_box_of_fg_points[:, -1].long()
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


    def generate_sa_center_ness_mask(self):
        sa_pos_mask = self.forward_ret_dict['target_dicts']['sa_ins_labels']
        sa_gt_boxes = self.forward_ret_dict['target_dicts']['sa_gt_box_of_fg_points']
        sa_xyz_coords = self.forward_ret_dict['target_dicts']['sa_xyz_coords']
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)):
            pos_mask = sa_pos_mask[i] > 0
            gt_boxes = sa_gt_boxes[i]
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:]
            xyz_coords = xyz_coords[pos_mask].clone().detach()
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3]
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

            template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
            margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
            distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
            distance[:, 1, :] = -1 * distance[:, 1, :]
            distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
            distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

            centerness = distance_min / distance_max
            centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
            centerness = torch.clamp(centerness, min=1e-6)
            centerness = torch.pow(centerness, 1/3)

            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
            centerness_mask[pos_mask] = centerness

            sa_centerness_mask.append(centerness_mask)
        return sa_centerness_mask

    def get_sa_ins_layer_loss(self, tb_dict=None):
        sa_ins_labels = self.forward_ret_dict['target_dicts']['sa_ins_labels']
        sa_ins_preds = self.forward_ret_dict['target_dicts']['sa_ins_preds']
        sa_centerness_mask = self.generate_sa_center_ness_mask()
        sa_ins_loss, ignore = 0, 0
        for i in range(len(sa_ins_labels)): # valid when i =1, 2
            if len(sa_ins_preds[i]) != 0:
                try:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, self.num_class)
                except:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, 1)

            else:
                ignore += 1
                continue
            point_cls_labels = sa_ins_labels[i].view(-1)
            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]

            if ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i+1][0]):
                centerness_mask = sa_centerness_mask[i]
                one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])

            point_loss_ins = self.ins_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()        
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_ins = point_loss_ins * loss_weights_dict.get('ins_aware_weight',[1]*len(sa_ins_labels))[i]

            sa_ins_loss += point_loss_ins
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({
                'sa%s_loss_ins' % str(i): point_loss_ins.item(),
                'sa%s_pos_num' % str(i): pos_normalizer.item()
            })

        sa_ins_loss = sa_ins_loss / (len(sa_ins_labels) - ignore)
        tb_dict.update({
                'sa_loss_ins': sa_ins_loss.item(),
            })
        return sa_ins_loss, tb_dict



    def get_sample_box_labels(self, input_dict):
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
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)

        if target_cfg.get('INS_AWARE_ASSIGN', False):
            sa_ins_labels, sa_gt_box_of_fg_points, sa_xyz_coords, sa_gt_box_of_points, sa_box_idxs_labels = [],[],[],[],[]
            sa_ins_preds = input_dict['sa_ins_preds']
            for i in range(1, len(sa_ins_preds)): # valid when i = 1,2 for IA-SSD
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



    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        # ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        if gt_boxes.shape[1] > 8:
            gt_boxes_pad = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 3))   # CHK MARK, no class, for iou calculation
        else:
            gt_boxes_pad = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1)) 
        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

            gt_boxes_pad[k] = gt_boxes[k, :7]

        return heatmap, ret_boxes, inds, mask, gt_boxes_pad

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'gt_boxes': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            gt_boxes_list = []  # CHK MARK, for iou need
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names): # filter gt_boxes
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]  # return an empty tensor
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, gt_boxes_pad = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_pad.to(gt_boxes_single_head))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))   # concat batch
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(torch.stack(gt_boxes_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):

        tb_dict = {}
        loss = 0

        if getattr(self, 'SEM_TASK', False):
            target_sem = self.forward_ret_dict['label_sem']
            pred_sem = self.forward_ret_dict['sem_pred']
            assert target_sem.shape[0] == pred_sem.shape[0]
            sem_loss = self.sem_criterion(pred_sem,target_sem)['loss']*self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['sem_weight']
            tb_dict['sem_loss'] = sem_loss.item()
            
            return sem_loss, tb_dict   
            
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx] # [4, 500, 8]
            gt_boxes = target_dicts['gt_boxes'][idx] # [4, 500, 7]
            mask = target_dicts['masks'][idx]
            ind = target_dicts['inds'][idx]

            # ususal box regression
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            reg_loss = self.reg_loss_func(pred_boxes, mask, ind, target_boxes)
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            # IoU based regression loss
            pred_boxes = centernet_utils.generate_dense_boxes(
                pred_dict=pred_dict,
                feature_map_stride=self.feature_map_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range)
            pred_boxes = torch.clamp(pred_boxes, min=-200., max=200.)   # avoid large number          

            # IoU prediction loss
            if pred_dict.get('iou', None) is not None:
                pred_boxes_for_iou = pred_boxes.detach()
                iou_loss = self.iou_loss_func(pred_dict['iou'], mask, ind, pred_boxes_for_iou, gt_boxes)
                iou_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
                tb_dict['iou_loss_%d' % idx] = iou_loss.item()
            else: iou_loss = 0

            loss += hm_loss + loc_loss + iou_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()

        # fake_sem_loss 
        if self.use_det_for_sem:
            target_sem = self.forward_ret_dict['label_sem']
            pred_sem = self.forward_ret_dict['sem_pred']
            
            if target_sem.shape[0] == pred_sem.shape[0]:

                car_mask = target_sem > 0
                car_sem = pred_sem[car_mask]
                target_sem = target_sem[car_mask]

                if car_sem.shape[0] == 0:
                    return loss, tb_dict

                assert target_sem.shape[0] == car_sem.shape[0]
                loss_ratio = car_sem.shape[0] / pred_sem.shape[0]
                sem_loss = self.sem_criterion(car_sem,target_sem)['loss'] * loss_ratio
                tb_dict['sem_loss'] = sem_loss.item()
                loss = loss + sem_loss
            else:
                print(f"Shape not match: target {target_sem.shape[0]} || pred {pred_sem.shape[0]}")
   
        # sa loss
        if target_dicts.get('sa_ins_labels', None) is not None:
            sa_loss_cls, tb_dict_0 = self.get_sa_ins_layer_loss()
            tb_dict.update(tb_dict_0)
            loss += sa_loss_cls

        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for i in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):    # for each head
            batch_box_preds = centernet_utils.generate_dense_boxes(
                pred_dict=pred_dict,
                feature_map_stride=self.feature_map_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            B, _, H, W = batch_box_preds.size()
            batch_box_preds = batch_box_preds.permute(0, 2, 3, 1).view(B, H*W, -1)
            batch_hm = pred_dict['hm'].sigmoid().permute(0, 2, 3, 1).view(B, H*W, -1)

            if 'iou' in pred_dict.keys():
                batch_iou = pred_dict['iou'].permute(0, 2, 3, 1).view(B, H*W)
                batch_iou = (batch_iou + 1) * 0.5
            else: batch_iou = torch.ones((B, H*W)).to(batch_hm.device)

            for i in range(B):   # for each batch
                box_preds = batch_box_preds[i]
                hm_preds = batch_hm[i]
                iou_preds = batch_iou[i]
                scores, labels = torch.max(hm_preds, dim=-1)
                labels = self.class_id_mapping_each_head[idx][labels.long()]

                if self.rectifier.size(0) > 1:           # class specific
                    assert self.rectifier.size(0) == self.num_class
                    rectifier = self.rectifier[labels]   # (H*W,)
                else: rectifier = self.rectifier

                score_mask = scores > post_process_cfg.SCORE_THRESH
                distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                              & (box_preds[..., :3] <= post_center_range[3:]).all(1)
                mask = distance_mask & score_mask

                iou_preds = torch.clamp(iou_preds, min=0, max=1.)
                scores = torch.pow(scores, 1 - rectifier) \
                       * torch.pow(iou_preds, rectifier)

                box_preds = box_preds[mask]
                scores = scores[mask]
                labels = labels[mask]

                if post_process_cfg.NMS_CONFIG.NMS_NAME == 'agnostic_nms':
                    selected, selected_scores = \
                            model_nms_utils.class_agnostic_nms(
                                box_scores=scores, box_preds=box_preds,
                                nms_config=post_process_cfg.NMS_CONFIG,
                    )
                    selected_boxes = box_preds[selected]
                    selected_labels = labels[selected]

                elif post_process_cfg.NMS_CONFIG.NMS_NAME == 'class_specific_nms':
                    selected_scores, selected_labels, selected_boxes = \
                            model_nms_utils.class_specific_nms(
                                cls_scores=scores, box_preds=box_preds, 
                                labels=labels, nms_config=post_process_cfg.NMS_CONFIG,
                                class_id=self.class_id_mapping_each_head[idx],
                    )
                else:
                    raise NotImplementedError
                    
                # selected_labels = self.class_id_mapping_each_head[idx][selected_labels.long()]

                ret_dict[i]['pred_boxes'].append(selected_boxes)
                ret_dict[i]['pred_scores'].append(selected_scores)
                ret_dict[i]['pred_labels'].append(selected_labels)

        for i in range(batch_size): # concat head results
            ret_dict[i]['pred_boxes'] = torch.cat(ret_dict[i]['pred_boxes'], dim=0)
            ret_dict[i]['pred_scores'] = torch.cat(ret_dict[i]['pred_scores'], dim=0)
            ret_dict[i]['pred_labels'] = torch.cat(ret_dict[i]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):

        data_dict['gt_boxes'] = data_dict['gt_boxes'][:,:,[0,1,2,3,4,5,6,-1]]

        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = [] # is a list actually
        for head in self.heads_list:
            pred_dicts.append(head(x))

        self.SEM_TASK = False
        if data_dict.get('label_sem', None) is not None:
            self.SEM_TASK = True
            self.forward_ret_dict['label_sem'] = data_dict['label_sem']
            self.forward_ret_dict['sem_pred'] = data_dict['sem_pred']

            # Save real car
            # car_mask = self.forward_ret_dict['label_sem'] == 3
            # car_points = data_dict["points"][car_mask]
            # np.savetxt("./real_car_points.txt",car_points.detach().cpu().numpy()[:,1:])

            return data_dict

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            if data_dict.get('sa_ins_preds', None) is not None:
                add_sample_dict = self.get_sample_box_labels(data_dict)
                target_dict.update(add_sample_dict)
                target_dict.update({
                    'sa_ins_preds': data_dict['sa_ins_preds']
                    })

            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if self.use_det_for_sem:
            self.forward_ret_dict['sem_pred'] = data_dict['sem_pred']
            self.forward_ret_dict['label_sem'] = data_dict['fake_sem_tags']

            # car_mask = self.forward_ret_dict['label_sem'] > 0
            # car_points = data_dict["points"][car_mask]
            # np.savetxt("./car_points.txt",car_points.detach().cpu().numpy()[:,1:])

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
