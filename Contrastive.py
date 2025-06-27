
import pdb
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import open3d as o3d

from util.dist_utils import L2Dist, ChamferDist, HausdorffDist
from util.augment import drop, rotation, scaling, shear, translation, jitter
from torchvision.transforms import Compose
from util.distri_loss import DistriLoss


def get_lr_scheduler(optim, scheduler, total_step):
    '''
    get lr values
    '''
    lrs = []
    for step in range(total_step):
        lr_current = optim.param_groups[0]['lr']
        lrs.append(lr_current)
        if scheduler is not None:
            scheduler.step()
    return lrs
# global
model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3))
initial_lr = 1.
total_step = 200


def grad_scores(logits, feature):
    grads = torch.autograd.grad(logits, feature, grad_outputs=torch.ones_like(logits), create_graph=True)
    feature_gradients = grads[0]
    feature_gradients_abs = torch.abs(feature_gradients) # 量化梯度对特征的影响程度，而忽略梯度的方向
    feature = feature_gradients_abs * feature

    return feature,feature_gradients_abs 


class CWContra:

    def __init__(self, model, dist_func_cd, dist_func_h, contra_func, fea_func, clip_func):

        self.model = model.cuda()
        self.model.eval()

        self.dist_func_cd = dist_func_cd
        self.dist_func_h = dist_func_h
        self.contra_func = contra_func
        self.curv_func = curv_func
        self.fea_func = fea_func
        self.clip_func = clip_func
        # self.functions = functions

    def attack(self, pc, target, args):

        B, K = pc.shape[:2]
        data = pc[:,:,:3].float().cuda().detach()
        normal = pc[:,:,-3:].float().cuda()
        data = data.transpose(1, 2).contiguous() # torch.Size([B, 3, 1024])
        normal = normal.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        ori_data1 = data.clone().detach()
        ori_data.requires_grad = False
        ori_data1.requires_grad_()
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()  # [B]

        logits_grad= self.model(ori_data1)
        weight = logits_grad[-1]

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))

        for binary_step in range(args.binary_step):
            offset = torch.zeros(B, 3, K).cuda()
            nn.init.normal_(offset, mean=0, std=1e-3) # 正态分布初始化

            adv_data = ori_data.clone() + offset
            adv_data.requires_grad_()
            opt = optim.Adam([adv_data], lr=args.attack_lr)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9990)          

            dist_loss_cd = torch.tensor(0.).cuda()
            dist_loss_h = torch.tensor(0.).cuda()
            contra_loss = torch.tensor(0.).cuda()
            fea_loss = torch.tensor(0.).cuda()

            total_time = 0.
            optimize_time = 0.
            clip_time = 0.
            update_time = 0.

            grad_t = torch.zeros_like(adv_data) 
            grad_t.cuda()
            # one step in binary search
            for iteration in range(args.num_iter):
                t1 = time.time()

                functions = [shear, translation, rotation, jitter, scaling, drop] # shear, translation, rotation, jitter, scaling, drop


                aug_pc_1 = []
                aug_pc_2 = []
                for i in range(B):
                    pc = ori_data[i:i+1]
                    selected_function1 = random.choice(functions)
                    selected_function2 = random.choice(functions)
                    pc1 = selected_function1(pc)
                    pc2 = selected_function2(pc)
                    aug_pc_1.append(pc1)
                    aug_pc_2.append(pc2)
                aug_pc1 = torch.cat(aug_pc_1, dim=0)
                aug_pc1.requires_grad = False
                aug_pc2 = torch.cat(aug_pc_2, dim=0)
                aug_pc2.requires_grad = False

                ori_logits= self.model(ori_data)
                if isinstance(ori_logits, tuple):
                    ori_features = ori_logits[1]
                    ori_logits = ori_logits[0]
                features_ori, _ = grad_scores(ori_logits, ori_features)

                aug_pc1_logits = self.model(aug_pc1)
                if isinstance(aug_pc1_logits, tuple):
                    aug_pc1_logits = aug_pc1_logits[0]


                aug_pc2_logits = self.model(aug_pc2)
                if isinstance(aug_pc2_logits, tuple):
                    aug_pc2_logits = aug_pc2_logits[0]
                    
                #original adversarial loss
                adv_pc_logits = self.model(adv_data)
                if isinstance(adv_pc_logits, tuple):
                    adv_features = adv_pc_logits[1]
                    adv_pc_logits = adv_pc_logits[0]
                features_adv,_ = grad_scores(adv_pc_logits, adv_features) 

                # 联合优化
                contra_loss = self.contra_func(adv_pc_logits, aug_pc1_logits, aug_pc2_logits).mean()

                fea_loss = self.fea_func(features_ori, features_adv).mean()

                dist_loss_cd = self.dist_func_cd(adv_data, ori_data)
                dist_loss_h = self.dist_func_h(adv_data, ori_data)
                dist_loss =args.Chamfer_loss_weight * dist_loss_cd + args.Hausdorff_loss_weight * dist_loss_h
                
                loss = args.contra_loss_weight * contra_loss + args.dist_loss_weight * dist_loss + args.fea_loss_weight * (1 / fea_loss)
                opt.zero_grad()

                loss.backward()

                opt.step()
                if args.is_use_lr_scheduler:
                    lr_scheduler.step()

                t2 = time.time()
                optimize_time += t2 - t1

                # 裁剪，将对抗点云裁剪到扰动范围内
                adv_data.data = self.clip_func(adv_data.detach().clone(), ori_data)

                t3 = time.time()
                clip_time += t3 - t2

                # print
                pred = torch.argmax(adv_pc_logits, dim=1)  # [B]
                success_num = (pred != target).sum().item()

                if iteration % 10 == 0 or iteration== 199:
                    print('Step {}, iteration {}, success {}/{}\n'
                            'contra_loss: {:.4f}, dist_loss: {:.4f}, curv_loss: {:.4f}, fea_loss: {:.4f}'.
                            format(binary_step, iteration, success_num, B, contra_loss.item(), 
                                    dist_loss.item(), curv_loss.item(), fea_loss.item()))

                # record values!
                dist_val = torch.sqrt(torch.sum(
                    (adv_data - ori_data) ** 2, dim=[1, 2])).\
                    detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                input_val = adv_data.detach().cpu().numpy()  # [B, 3, K]

                # update
                for e, (dist, pred, label, ii) in \
                        enumerate(zip(dist_val, pred_val, label_val, input_val)):
                    if dist < o_bestdist[e] and pred != label and args.contra_loss_weight < 0.001:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                t4 = time.time()
                update_time += t4 - t3

                total_time += t4 - t1

                if iteration % 10 == 0:
                    print('total time: {:.2f}, opt: {:.2f}, '
                            'clip: {:.2f}, update: {:.2f}'.
                            format(total_time, optimize_time,
                                    clip_time, update_time))
                    total_time = 0.
                    optimize_time = 0.
                    clip_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

        # end of CW attack
        # fail to attack some examples
        fail_idx = (o_bestscore < 0)
        o_bestattack[fail_idx] = input_val[fail_idx]

        adv_pc = torch.tensor(o_bestattack).to(adv_data)
        adv_pc = self.clip_func(adv_pc, ori_data)

        logits = self.model(adv_pc)
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = torch.argmax(logits, dim=-1)
        
        ori_logits = self.model(ori_data)
        if isinstance(ori_logits, tuple):
            ori_logits = ori_logits[0]
        ori_preds = torch.argmax(ori_logits, dim=-1)
        
        # return final results
        ori_class_accuracy_num = (ori_preds == target).sum().item()
        success_num = (preds != target).sum().item()
        print('Successfully attack {}/{}'.format(success_num, B))
        
        return o_bestdist, adv_pc.detach().cpu().numpy().transpose((0, 2, 1)), ori_class_accuracy_num, success_num
