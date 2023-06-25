import torch
import numpy as np
import torch.nn as nn
from ..unets import U_Net
from .projection import Projection


class Classifier(nn.Module):
    def __init__(self, input_channels, layers, sem_class):
        super(Classifier, self).__init__()
        self.later_list = layers
        self.act = nn.ReLU()
        self.shared_mlps = []
        self.input_channels = input_channels
        for dim in self.later_list:
            self.shared_mlps.extend([
            nn.Linear(input_channels, dim,bias=False),
            nn.ReLU(),
            nn.Dropout(0.2)
            ])
            input_channels = dim
        self.shared_mlps.extend([
            nn.Linear(dim, sem_class, bias=False)
            ])
        self.classifier = nn.Sequential(*self.shared_mlps)

    def forward(self, input_features):
        assert input_features.shape[-1] == self.input_channels
        return self.classifier(input_features)


class MLTSSD_encoding(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.mlp_list = self.model_cfg.MLPS
        self.sem_mlp_list = self.model_cfg.SEM_MLPS
        self.det_mlp_list = self.model_cfg.DET_MLPS
        self.input_channel = 4

        self.mlps = nn.ModuleList()
        self.sem_mlps = nn.ModuleList()
        self.det_mlps = nn.ModuleList()
        
        # self.num_class = kwargs['num_class']
        self.sem_num_class = model_cfg.SEM_CLASS_NUM
        self.npoint = model_cfg.NPOINT
        self.point_cloud_range = np.array(self.model_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(model_cfg.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        bev_shape = self.grid_size[:2]

        self.shared_mlps = []
        input_channel = self.input_channel
        for dim in self.mlp_list:
            self.shared_mlps.extend([
                nn.Linear(input_channel, dim,bias=False),
                nn.ReLU()
            ])
            input_channel = dim
        self.mlps.append(nn.Sequential(*self.shared_mlps))

        self.shared_sem_mlps = []
        input_channel = self.input_channel
        for dim in self.sem_mlp_list:
            self.shared_sem_mlps.extend([
                nn.Linear(input_channel, dim,bias=False),
                nn.ReLU()
            ])
            input_channel = dim
        self.sem_mlps.append(nn.Sequential(*self.shared_sem_mlps))

        self.shared_det_mlps = []
        input_channel = self.input_channel
        for dim in self.det_mlp_list:
            self.shared_det_mlps.extend([
                nn.Linear(input_channel, dim,bias=False),
                nn.ReLU()
            ])
            input_channel = dim
        self.det_mlps.append(nn.Sequential(*self.shared_det_mlps))


        self.proj = Projection(pc_range = model_cfg.POINT_CLOUD_RANGE, bev_shape = bev_shape)
        self.num_bev_features = self.mlp_list[-1]
        self.encoder = U_Net(in_ch=self.sem_mlp_list[-1], out_ch=self.sem_mlp_list[-1])

        self.classifier = Classifier(input_channels=self.mlp_list[-1]*2, layers=model_cfg.CLASSIFIER, sem_class=self.sem_num_class)

 
    def cosine_similarity(self,x,y):
        num = torch.matmul(x,y.T).squeeze()
        denom = torch.norm(x, dim = -1) * torch.norm(y.squeeze())
        return num / denom


    def forward(self, batch_dict):
        # visible points
        # vs_points = []

        batch_size = batch_dict['batch_size']
        coord = batch_dict['points'][:,:4]
        origin_pw_feature = batch_dict['points'][:,1:]
        assert origin_pw_feature.shape[-1] == 4

        bone_pw_feature = origin_pw_feature.clone()
        for mlp in self.mlps:
            pw_feature = mlp(bone_pw_feature)
            bone_pw_feature = pw_feature

        sem_pw_feature = origin_pw_feature.clone()
        for mlp in self.sem_mlps:
            pw_feature = mlp(sem_pw_feature)
            sem_pw_feature = pw_feature

        det_pw_feature = origin_pw_feature.clone()
        for mlp in self.det_mlps:
            pw_feature = mlp(det_pw_feature)
            det_pw_feature = pw_feature

        keep_bev = self.proj.init_bev_coord(coord)[1]
        init_bev = self.proj.p2g_bev(bone_pw_feature[keep_bev], batch_size)

        # U-Net to learning deeper features
        output_bev = self.encoder(init_bev)
        # new point-wise features
        bone_pw_feature = self.proj.g2p_bev(output_bev)

        c_bev = bone_pw_feature.shape[1]
        cmplt_pw_feature = output_bev.new_zeros([coord.shape[0], c_bev])
        cmplt_pw_feature[keep_bev] = bone_pw_feature # Only change features in range
        cmplt_sem_pw_feature = torch.cat([cmplt_pw_feature, sem_pw_feature], dim = -1)
        li_sem_pred = self.classifier(cmplt_sem_pw_feature)
        # cmplt_det_pw_feature = torch.cat([cmplt_pw_feature[:,-4:], det_pw_feature], dim = -1)
        cmplt_det_pw_feature = det_pw_feature.clone()

        # kitti
        new_points = []
        new_features = []
        soft_bg_points = []
        for batch_idx in range(batch_size):
            batch_mask = coord[:,0] == batch_idx
            batch_points = batch_dict['points'][batch_mask]
            batch_features = cmplt_det_pw_feature[batch_mask]
            # if batch_idx == 0:
            #     vs_points.append(torch.cat([batch_points[:,1:4],batch_dict['fake_labels'][batch_mask].view(-1,1)], dim = -1))
            #     vs_points.append(torch.cat([batch_points[:,1:4],torch.argmax(li_sem_pred, dim = -1)[batch_mask].view(-1,1)], dim = -1))

            if batch_points.shape[0] <= self.npoint:
                emb_points = batch_points.new_zeros([self.npoint, batch_points.shape[-1]])
                emb_features = batch_points.new_zeros([self.npoint, batch_features.shape[-1]])
                emb_points[:,0] = batch_idx
                emb_points[:batch_points.shape[0],:] = batch_points
                emb_features[:batch_points.shape[0],:] = batch_features
                new_points.append(emb_points)
                new_features.append(emb_features)

                # Add soft_bg_points in each batch
                batch_sem_pred = li_sem_pred[batch_mask]
                batch_sem_args = torch.argmax(batch_sem_pred, dim = -1)
                fg_tag1 = batch_sem_args > 0
                fg_tag2 = batch_sem_args < 11 # Nuscenes
                fg_tag = fg_tag1 & fg_tag2
                soft_bg_points.append(batch_points[fg_tag]) # kitti
            else:
                batch_sem_pred = li_sem_pred[batch_mask]
                batch_sem_args = torch.argmax(li_sem_pred[batch_mask], dim = -1)
                fg_tag1 = batch_sem_args > 0
                fg_tag2 = batch_sem_args < 11 # Nuscenes
                fg_tag = fg_tag1 & fg_tag2

                # Add soft_bg_points in each batch
                soft_bg_points.append(batch_points[fg_tag])

                if torch.sum(fg_tag) >= self.npoint:
                    batch_points = batch_points[fg_tag]
                    batch_features = batch_features[fg_tag]
                    batch_sem_pred = batch_sem_pred[fg_tag][:,1:]
                    cls_features_max, class_pred = batch_sem_pred.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    _, sample_idx = torch.topk(score_pred, self.npoint, dim=-1) 
                    new_points.append(batch_points[sample_idx])
                    new_features.append(batch_features[sample_idx])
                else:
                    last_npoint = self.npoint - torch.sum(fg_tag)
                    fg_points = batch_points[fg_tag]
                    fp_features = batch_features[fg_tag]
                    
                    bg_points = batch_points[~fg_tag]
                    bg_features = batch_features[~fg_tag]
                    # batch_bg_sem_pred = batch_sem_pred[~fg_tag]
                    # if batch_dict['gt_box'].shape[-1] < 9: # kitti
                    #     abs_bg = batch_points.new_zeros([1,self.sem_num_class + 1])
                    # else:
                    #     abs_bg = batch_points.new_zeros([1,self.sem_num_class])
                    # abs_bg[0,0] = 1
                    # abs_cos_features = self.cosine_similarity(torch.sigmoid(batch_bg_sem_pred), abs_bg)
                    # _, sample_idx = torch.topk(-abs_cos_features, last_npoint, dim=-1) 
# 
                    
                    sample_idx = np.random.permutation(bg_points.shape[0])[:last_npoint]
                    _soft_bg_points = bg_points[sample_idx]
                    soft_bg_features = bg_features[sample_idx]

                    batch_points = torch.cat([fg_points, _soft_bg_points], dim = 0)
                    batch_features = torch.cat([fp_features, soft_bg_features], dim = 0)

                    assert batch_points.shape[0] == self.npoint
                    new_points.append(batch_points)
                    new_features.append(batch_features)

        # vs_points.append(new_points[0][:,1:4])
        points = torch.cat(new_points, dim = 0)
        new_features = torch.cat(new_features, dim = 0)

        # soft_bg_points = torch.cat(soft_bg_points, dim = 0)

    
        batch_dict.update({
            'features': new_features,
            'points': points,
            'sem_pred': li_sem_pred,
            'soft_bg_points': soft_bg_points,
            # 'vs_points': vs_points
        })
        #

        # batch_dict['li_cls_pred'] = li_cls_pred

        return batch_dict




        