from tqdm import tqdm
import argparse
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("/home/chenhai-fwxz/Point-Transformers-master/benchmark_pc_attack/baselines")
from dataset import ModelNet40Attack, ModelNet40Transfer, load_data
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from model.Hengshuang.model import PointTransformerCls
from model.PCT.model import Pct
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed

from config import BEST_WEIGHTS
from config import MAX_TEST_BATCH as BATCH_SIZE

from plyfile import PlyData,PlyElement
import pickle


def test_normal(model_name):
    """Normal test mode.
    Test on all data.
    """
    trans_model.eval()
    at_num, at_denom = 0, 0  #

    num, denom = 0, 0 #
    num_error = 0
    with torch.no_grad():
        for ori_data, adv_data, label in test_loader:
            ori_data, adv_data, label = \
                ori_data.float().cuda(), adv_data.float().cuda(), label.long().cuda()
            # to [B, 3, N] point cloud
            ori_data = ori_data.transpose(1, 2).contiguous()
            adv_data = adv_data.transpose(1, 2).contiguous()
            batch_size = label.size(0)
            # batch in
            if model_name.lower() == 'pointtransformer':
                ori_data = ori_data.transpose(1, 2).contiguous() #[B,N,3]
                logits,_ = trans_model(ori_data)
                adv_data = adv_data.transpose(1, 2).contiguous() #[B,N,3]
                adv_logits,_ = trans_model(adv_data)
            else:
                logits = trans_model(ori_data)
                logits = logits[0]
                adv_logits = trans_model(adv_data)
                adv_logits = adv_logits[0]
            ori_preds = torch.argmax(logits, dim=-1)
            adv_preds = torch.argmax(adv_logits, dim=-1)
            mask_ori = (ori_preds == label)
            mask_adv = (adv_preds == label)
            err_num = (adv_preds != label)
            at_denom += mask_ori.sum().float().item() # 分类成功
            at_num += mask_ori.sum().float().item() - (mask_ori * mask_adv).sum().float().item() # 分类成功的样本生成的对抗样本分类不成功
            denom += float(batch_size)
            num_error += err_num.sum().float().item()
            num += mask_adv.sum().float()

    print('Overall attack success rate: {:.4f}'.format(at_num / (at_denom + 1e-9)))
    # ASR = at_num / (at_denom + 1e-9)
    print('Overall accuracy: {:.4f}'.format(at_denom / (denom + 1e-9))) # 模型本身的分类准确率
    print('top-1 error:{:.4f}'.format(num_error/(denom + 1e-9)))
    # print(ASR)
 

if __name__ == "__main__":
    # Training settings
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_path', type=str,
                        default='')
    parser.add_argument('--trans_model', type=str, default='',
                        choices=['pointtransformer','pct', 'pointnet','pointnet2','dgcnn','pointconv'],
                        help='Model to use, [pointtransformer,pointnet, pointnet++, dgcnn]') #攻击模型
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40', 'aug_mn40'])
    parser.add_argument('--normalize_pc', type=str2bool, default=False,
                        help='normalize in dataloader')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='Size of batch, use config if not specified')
    #PT参数
    parser.add_argument('--model_name', default='Hengshuang', help='model name')
    parser.add_argument('--nneighbor', type=int, default=8)
    parser.add_argument('--nblocks', type=int, default=4)
    parser.add_argument('--transformer_dim', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=40)
    #
    #PCT参数
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
    #
    parser.add_argument('--budget', type=float, default=0.45, 
                        help='FGM attack budget') #扰动大小 0.01 0.18 0.45
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()

    BATCH_SIZE = BATCH_SIZE[args.num_points]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.trans_model]
    set_seed(1)
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # build trans model
    if args.trans_model.lower() == 'pointtransformer':
        trans_model = PointTransformerCls(args) 
    elif args.trans_model.lower() == 'pct':
        trans_model = Pct(args) # 模型输入[B,3,N]
    elif args.trans_model.lower() == 'dgcnn':
        trans_model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.trans_model.lower() == 'pointnet':
        trans_model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.trans_model.lower() == 'pointnet2':
        trans_model = PointNet2ClsSsg(num_classes=40)
    elif args.trans_model.lower() == 'pointconv':
        trans_model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)
        
    trans_state_dict = torch.load(
        BEST_WEIGHTS[args.trans_model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.trans_model]))
    if args.trans_model.lower() == 'pointtransformer':
        trans_model.load_state_dict(trans_state_dict['model_state_dict'])
    else:
        try:
            trans_model.load_state_dict(trans_state_dict)
        except RuntimeError:
            # eliminate 'module.' in keys
            trans_state_dict = {k[7:]: v for k, v in trans_state_dict.items()}
            trans_model.load_state_dict(trans_state_dict)
    trans_model = trans_model.cuda()
    # prepare data
    # data_path = os.path.join(args.data_root, args.prefix)
    test_set = ModelNet40Transfer(args.data_path, num_points=args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=8,
                             pin_memory=True, drop_last=True)
    data_num = len(test_set)
    print(args.data_path)

    test_normal(args.trans_model)