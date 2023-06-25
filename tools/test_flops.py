import argparse
import torch
import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./cfgs/kitti_models/IA-SSD.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='../output/cfgs/kitti_models/IA-SSD/default/ckpt/checkpoint_epoch_80.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--item', type=int, default=200, help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def input_constructor(input_shape):
    batch_dict = {}
    points = np.random.randn(input_shape[0], input_shape[1])
    batch_idx = np.zeros([points.shape[0],1])
    points = np.hstack([batch_idx,points])
    batch_dict.update(
        {
        'points':points,
        'batch_size': 1
        }
        )
    return batch_dict


def main():
    import pcdet.datasets as DT
    from pcdet.utils.flops_counter import get_model_complexity_info
    args, cfg = parse_config()
    logger = common_utils.create_logger(log_file = './flops_result.txt')
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    demo_dataset = getattr(DT,cfg.DATA_CONFIG.DATASET)(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False,
        logger=logger,
    )
    
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    data_dict = demo_dataset.__getitem__(args.item)
    input_shape = data_dict['points'].shape
    data_dict = demo_dataset.collate_batch([data_dict])
    
    if torch.cuda.is_available():
        load_data_to_gpu(data_dict)

    flops, params = get_model_complexity_info(model, input_shape, input_dict=data_dict, input_constructor=input_constructor)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')





    logger.info('Demo done.')

if __name__ == '__main__':
    main()
