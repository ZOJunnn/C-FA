"""Targeted point perturbation attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import sys
sys.path.append('../')
sys.path.append('./')

from config import BEST_WEIGHTS
from config import MAX_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack, ModelNetDataLoader
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from model.PCT.model import Pct
from util.utils import str2bool, set_seed

from plyfile import PlyData,PlyElement

from util.augment import drop, rotation, scaling, shear, translation, jitter


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='/home/chenhai-fwxz/ZYJ/modelnet40_normal_resampled')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40', 'aug_mn40'])
    parser.add_argument('--batch_size', type=int, default=10, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
        
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)


    # build model
    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    elif args.model.lower() == 'pct':
        model = Pct(args) # 模型输入[B,3,N]
    else:
        print('Model not recognized')
        exit(-1)

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = model.cuda()
    model.eval()

    # setup attack settings

    test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                    shuffle=False, num_workers=4, 
                    drop_last=False)
    
    data_num = len(test_set)
    num = 0
    for pc, label in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), label.long().cuda(non_blocking=True)

        pc = pc.transpose(1, 2).contiguous() #[B, 3, N]
            
        # pc1 = rotation(pc) # 全局旋转，沿XYZ轴
        # pc1 = translation(pc) # 随机平移，沿XYZ轴
        # pc1 = flip(pc) # 随机翻转，沿xoy平面
        pc1 = scaling(pc) # 全局缩放
        # pc1 = drop(pc) # 随机丢点
        # pc1 = shear(pc) # 随机裁剪
        # pc1 = jitter(pc)
        logits = model(pc1)
        if isinstance(logits, tuple):  # PointNet
            logits = logits[0]
        pred = torch.argmax(logits, dim=1)  # [B]
        success_num = (pred == label).sum().item()
        num += success_num
    accuracy = num/data_num
    print("{}/{}".format(args.model, accuracy))