# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from pathlib import Path
from tqdm import tqdm

import mmcv
import mmengine
import torch
from mmengine import DictAction

from mmpretrain.datasets import build_dataset
from mmpretrain.structures import DataSample
from mmpretrain.visualization import UniversalVisualizer


def _resize_img(results: dict):
    """Resize images with ``results['scale']``."""

    img, w_scale, h_scale = mmcv.imresize(
        results['img'],
        results['scale'],
        # interpolation=self.interpolation,
        return_scale=True,
        # backend=self.backend
        )
    results['img'] = img
    results['img_shape'] = img.shape[:2]
    results['scale'] = img.shape[:2][::-1]
    results['scale_factor'] = (w_scale, h_scale)

def transform(**results) :
    """Transform function to resize images.

    Args:
        results (dict): Result dict from loading pipeline.

    Returns:
        dict: Resized results, 'img', 'scale', 'scale_factor',
        'img_shape' keys are updated in result dict.
    """
    assert 'img' in results, 'No `img` field in the input.'

    h, w = results['img'].shape[:2]
    if any([
            # conditions to resize the width
            results['edge'] == 'short' and w < h,
            results['edge'] == 'long' and w > h,
            results['edge'] == 'width',
    ]):
        width = results['scale']
        height = int(results['scale'] * h / w)
    else:
        height = results['scale']
        width = int(results['scale'] * w / h)
    results['scale'] = (width, height)

    _resize_img(results)
    return results['img']


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPreTrain evaluate prediction success/fail')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('result', help='test result json/pkl file')
    parser.add_argument(
        '--out-dir', required=True, help='dir to store output files')
    parser.add_argument(
        '--topk',
        default=300,
        type=int,
        help='Number of images to select for success/fail')
    parser.add_argument(
        '--dataset',
        choices=['test', 'val', 'train'],
        default='test',
        help='data set')
    parser.add_argument(
        '--resize',
        type=int,
        default=448,
        help='resize to store')
    parser.add_argument(
        '--rescale-factor',
        '-r',
        type=float,
        help='image rescale factor, which is useful if the output is too '
        'large or too small.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def save_imgs(result_dir, folder_name, results, dataset, resize=None, rescale_factor=None):
    full_dir = osp.join(result_dir, folder_name)
    vis = UniversalVisualizer()
    vis.dataset_meta = {'classes': dataset.CLASSES}

    # save imgs
    dump_infos = []
    for data_sample in tqdm(results):
        data_info = dataset.get_data_info(data_sample.sample_idx)
        if 'img' in data_info:
            img = data_info['img']
            name = str(data_sample.sample_idx)
        elif 'img_path' in data_info:
            img = mmcv.imread(data_info['img_path'], channel_order='rgb')
            name = Path(data_info['img_path']).name
        else:
            raise ValueError('Cannot load images from the dataset infos.')

        pred = data_sample.pred_label
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        gt = data_sample.gt_label
        if isinstance(gt, torch.Tensor):
            gt = gt.item()

        pred_name = dataset.CLASSES[pred]
        gt_name = dataset.CLASSES[gt]

        prefix = f'{gt_name}-》{pred_name}/'
        name = prefix + data_info['img_path'][5:]

        name = Path(name).with_suffix('.jpg')
        print(name)

        if rescale_factor is not None:
            img = mmcv.imrescale(img, rescale_factor)
        if resize is not None:
            img = transform(img=img, scale=int(resize), edge="short")
        vis.visualize_cls(
            img, data_sample, out_file=osp.join(full_dir, name))

        dump = dict()
        for k, v in data_sample.items():
            if isinstance(v, torch.Tensor):
                dump[k] = v.tolist()
            else:
                dump[k] = v
        if 'img_path' in data_info:
            dump['img_path'] = data_info['img_path']
        dump_infos.append(dump)

    mmengine.dump(dump_infos, osp.join(full_dir, folder_name + '.json'), ensure_ascii=False)


def main():
    args = parse_args()

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the dataloader
    if args.dataset == 'train':
        cfg.test_dataloader.dataset.ann_file = cfg.train_dataloader.dataset.ann_file
        print(cfg.test_dataloader.dataset.ann_file)
    cfg.test_dataloader.dataset.pipeline = []
    dataset = build_dataset(cfg.test_dataloader.dataset)

    results = list()
    for result in tqdm(mmengine.load(args.result)):
        data_sample = DataSample()
        data_sample.set_metainfo({'sample_idx': result['sample_idx']})
        data_sample.set_gt_label(result['gt_label'])
        data_sample.set_pred_label(result['pred_label'])
        data_sample.set_pred_score(result['pred_score'])
        results.append(data_sample)

    print(len(results))
    # sort result
    results = sorted(results, key=lambda x: torch.max(x.pred_score))

    success = list()
    # success_conf = list()
    fail = list()
    # fail_conf = list()
    for data_sample in tqdm(results):
        if (data_sample.pred_label == data_sample.gt_label).all():
            success.append(data_sample)
            # success_conf.append(max(data_sample.pred_score))
        else:
            fail.append(data_sample)
            # fail_conf.append(max(data_sample.pred_score))

    success = success[:args.topk]
    # fail = fail[-args.topk:]
    # topk_idx = sorted(range(len(success_conf)), key=lambda i: success_conf[i], reverse=True)[:args.topk]
    # success = [success[i] for i in topk_idx]
    # topk_idx = sorted(range(len(fail_conf)), key=lambda i: fail_conf[i], reverse=True)[:args.topk]
    # fail = [fail[i] for i in topk_idx]

    save_imgs(args.out_dir, 'success', success, dataset, args.resize, args.rescale_factor)
    save_imgs(args.out_dir, 'fail', fail, dataset, args.resize, args.rescale_factor)


if __name__ == '__main__':
    main()
