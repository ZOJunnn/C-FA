"""Apply baseline defense methods"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tqdm
import argparse
import numpy as np

import torch

from defense import SRSDefense, SORDefense, DUPNet
from config import PU_NET_WEIGHT


def defend(data_root, one_defense):
    # save defense result
    sub_roots = data_root.split('/')  # 分割data_root保存在列表sub_roots
    filename = sub_roots[-1]  # 获取data_root路径中的最后一个文件或目录的名称
    data_folder = data_root[:data_root.rindex(filename)] # 找到 filename的索引，获取data_root路径中最后一个文件或目录的上层目录路径
    save_folder = os.path.join(data_folder, one_defense)
    save_name = '{}_{}'.format(one_defense, filename)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # data to defend
    batch_size = 1 #128
    npz_data = np.load(data_root)   #ori_pc=ori_data.astype(np.float32),test_pc=attacked_data.astype(np.float32),test_label=real_label.astype(np.uint8))
    test_pc = npz_data['test_pc']  #(2468,1024,3)
    test_label = npz_data['test_label']#(2468,)
    try:
        target_label = npz_data['target_label']
    except KeyError:
        target_label = None
    try:
        ori_pc = npz_data['ori_pc'] #[2468,1024,3]
    except KeyError:
        ori_pc = None
    # target_label = npz_data['target_label']

    # defense module
    if one_defense.lower() == 'srs':
        defense_module = SRSDefense(drop_num=args.srs_drop_num)
    elif one_defense.lower() == 'sor':
        defense_module = SORDefense(k=args.sor_k, alpha=args.sor_alpha)
    elif one_defense.lower() == 'dup':
        up_ratio = 4
        defense_module = DUPNet(sor_k=args.sor_k,
                                sor_alpha=args.sor_alpha,
                                npoint=1024, up_ratio=up_ratio)
        defense_module.pu_net.load_state_dict(
            torch.load(PU_NET_WEIGHT))
        defense_module.pu_net = defense_module.pu_net.cuda()

    # defend
    all_defend_pc = []
    for batch_idx in tqdm.trange(0, len(test_pc), batch_size):   # tqdm.trange()函数用于在循环中显示进度条
        batch_pc = test_pc[batch_idx:batch_idx + batch_size]  # 获取一个batch_size的点云 [128,1024,3]
        batch_pc = torch.from_numpy(batch_pc)[..., :3]
        batch_pc = batch_pc.float().cuda()
        defend_batch_pc = defense_module(batch_pc)  # 防御处理

        # sor processed results have different number of points in each
        if isinstance(defend_batch_pc, list) or \
                isinstance(defend_batch_pc, tuple): # 检查defend_batch_pc是否为列表或元组
            defend_batch_pc = [
                pc.detach().cpu().numpy().astype(np.float32) for
                pc in defend_batch_pc    # 将defend_batch_pc中的每个元素转换为numpy数组
            ]
        else:
            defend_batch_pc = defend_batch_pc.\
                detach().cpu().numpy().astype(np.float32)
            defend_batch_pc = [pc for pc in defend_batch_pc]  #将defend_batch_pc中的每个元素以列表形式重新赋值给defend_batch_pc

        all_defend_pc += defend_batch_pc

    all_defend_pc = np.array(all_defend_pc)
    if target_label is None:
        if ori_pc is None:
            np.savez(os.path.join(save_folder, save_name),
                    test_pc=all_defend_pc,
                    test_label=test_label.astype(np.uint8))
        else:
            np.savez(os.path.join(save_folder, save_name),
                    ori_pc=ori_pc.astype(np.float32),
                    test_pc=all_defend_pc,
                    test_label=test_label.astype(np.uint8))
    else:
        np.savez(os.path.join(save_folder, save_name),
                 test_pc=all_defend_pc,
                 test_label=test_label.astype(np.uint8),
                 target_label=target_label.astype(np.uint8))
    print('defense result saved to {}'.format(os.path.join(save_folder, save_name)))
    # np.savez(os.path.join(save_folder, save_name),
    #          test_pc=all_defend_pc,
    #          test_label=test_label.astype(np.uint8),
    #          target_label=target_label.astype(np.uint8))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='/home/chenhai-fwxz/ZYJ/C-FA/new/results/1.0contra+0.3fea+1.0dist+1.0curv/pct/budget-0.45_t-0.15_success-0.9546.npz',
                        help='the npz data to defend')
    parser.add_argument('--defense', type=str, default='',
                        choices=['', 'srs', 'sor', 'dup'],
                        help='Defense method for input processing, '
                             'apply all if not specified')
    parser.add_argument('--srs_drop_num', type=int, default=500,
                        help='Number of point dropping in SRS') # 随即丢弃点数
    parser.add_argument('--sor_k', type=int, default=2,
                        help='KNN in SOR')
    parser.add_argument('--sor_alpha', type=float, default=1.1,
                        help='Threshold = mean + alpha * std')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # defense method
    if args.defense == '':
        all_defense = ['srs', 'sor', 'dup']
    else:
        all_defense = [args.defense]

    # apply defense
    for one_defense in all_defense:
        print('{} defense'.format(one_defense))
        # if data_root is a folder
        # then apply defense to all the npz file in it
        if os.path.isdir(args.data_root):  # 判断是否是目录
            all_files = os.listdir(args.data_root)  
            for one_file in all_files:   # 循环遍历all_files列表中的每个文件和子目录，将当前迭代的项赋值给one_file变量
                data_path = os.path.join(args.data_root, one_file)
                if os.path.isfile(data_path):  # 判断是文件
                    defend(data_path, one_defense=one_defense)
        else:
            defend(args.data_root, one_defense=one_defense)
