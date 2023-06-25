import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False



import numpy as np
import torch

from pcdet.config import cfg, cfg2, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_dataset import DemoDatasetV2, KittiDataset


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/PAGNet.yaml',
                        help='specify the config for demo')
    parser.add_argument('--surface_cfg_file', type=str, default='../surface_uncertainty/cfgs/sf_unc.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/kitti/training/velodyne/000001.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='../output/cfgs/kitti_models/IA-SSD/default/ckpt/checkpoint_epoch_80.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--item', type=int, default=200, help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def surface_parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default='../surface_uncertainty/cfgs/sf_unc.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='../surface_uncertainty/output/cfgs/sf_unc/V2/ckpt/checkpoint_epoch_400.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg2)

    return args, cfg2



def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


def liang():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    demo_dataset = DemoDatasetV2(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=None, logger=logger)

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():

        data_dict = demo_dataset.getitem(args.item)
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)

        V.draw_scenes(
            points=data_dict['points'][:, 1:],gt_boxes=data_dict['gt_boxes'][0], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        )

        if not OPEN3D_FLAG:
            mlab.show(stop=True)

    logger.info('Demo done.')


def surface():
    from surface_uncertainty.model import Generate_center

    args, cfg = parse_config()
    surface_args, surface_cfg = surface_parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    demo_dataset = DemoDatasetV2(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=None, logger=logger)

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # Generator
    generator = Generate_center(surface_cfg.MODEL)
    generator.load_params_from_file(filename=surface_args.ckpt, logger=logger, to_cpu=True)
    generator.cuda()
    generator.eval()

    
    with torch.no_grad():

        data_dict = demo_dataset.getitem(11)
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)
        surface_dict = generator.forward(data_dict)

        # V.draw_scenes(
        #     points=data_dict['points'][:, 1:],gt_boxes=data_dict['gt_boxes'][0], ref_boxes=pred_dicts[0]['pred_boxes'],
        #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        # )

        # V.draw_scenes(
        #     points=surface_dict['center_pred'][0,...][surface_dict['topk']],gt_boxes=data_dict['gt_boxes'][0], ref_boxes=pred_dicts[0]['pred_boxes'],
        #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        # )
        V.draw_surface_scenes(
            points=surface_dict['sa_points'][0,...],
            center_pred=surface_dict['center_pred'][0,...],
            topk=surface_dict['topk'],
            gt_boxes=data_dict['gt_boxes'][0], 
            ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], 
            ref_labels=pred_dicts[0]['pred_labels']
        )

        if not OPEN3D_FLAG:
            mlab.show(stop=True)

    logger.info('Demo done.')

def mmdet_vis():
    from visual_utils.visualizer.show_result import show_result, show_multi_modality_result
    # from pcdet.core.bbox import LiDARInstance3DBoxes
    from mmdet3d.core.bbox import LiDARInstance3DBoxes, Coord3DMode, Box3DMode
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    demo_dataset = KittiDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=True,
        root_path=None, logger=logger)

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    ############################################################
    for item in range(500):
    ############################################################
        with torch.no_grad():

            data_dict = demo_dataset.vis_mmdet_item(item)
            if isinstance(data_dict['gt_boxes'], np.ndarray):
                gt_boxes = data_dict['gt_boxes']
                if gt_boxes.shape[0] < 10:
                    continue

            logger.info(f"Get one in {item}! There are {gt_boxes.shape[0]} objects.")

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            points = data_dict['points'][:, 1:].cpu().numpy()
            
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                Coord3DMode.DEPTH)

            gt_boxes = data_dict['gt_boxes'].view(-1,8)[:,:-1].cpu().numpy()
            show_gt_bboxes = Box3DMode.convert(gt_boxes, Box3DMode.LIDAR,
                                                Box3DMode.DEPTH)

            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            show_pred_bboxes = Box3DMode.convert(pred_boxes, Box3DMode.LIDAR,
                                                    Box3DMode.DEPTH)

            file_name = str(data_dict['frame_id'][0])

            show_result(points,gt_bboxes=show_gt_bboxes,pred_bboxes=show_pred_bboxes,out_dir='./mmdet_vis',filename=file_name,show=False)
            
            show_calib = True
            if show_calib:
                img = data_dict['images']
                img = img.squeeze().cpu().numpy()
                
                #################################################################
                img_metas = data_dict['trans_lidar_to_cam'].squeeze().cpu().numpy()
                P2 = data_dict['trans_cam_to_img'].squeeze().cpu().numpy()
                P2 = np.vstack((P2, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
                img_metas = P2 @ img_metas
                #################################################################

                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_boxes, origin=(0.5, 0.5, 0.5))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_boxes, origin=(0.5, 0.5, 0.5))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas,
                    './mmdet_vis',
                    file_name,
                    box_mode='lidar',
                    show=False)





    logger.info('Demo done.')

if __name__ == '__main__':
    mmdet_vis()
