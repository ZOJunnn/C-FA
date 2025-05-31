import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------利用余弦相似度计算-------------------------------------------------
class simCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(simCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, adv_pc_logits, aug_pc1_logits, aug_pc2_logits):
        # Normalize the features
        aug_pc1_logits = F.normalize(aug_pc1_logits, p=2, dim=1)
        aug_pc2_logits = F.normalize(aug_pc2_logits, p=2, dim=1)
        adv_pc_logits = F.normalize(adv_pc_logits, p=2, dim=1)

        # 计算相似矩阵对角线上的相似度 经过F.normalize(p=2,dim=1)平方和开根号为1(余弦相似度)
        q_k = torch.mm(aug_pc1_logits, aug_pc2_logits.t()) 
        q_i_1 = torch.mm(aug_pc1_logits, adv_pc_logits.t())
        q_i_2 = torch.mm(aug_pc2_logits, adv_pc_logits.t())

        q_k_diag = torch.diag(q_k).unsqueeze(1)
        q_i_1_diag = torch.diag(q_i_1).unsqueeze(1)
        q_i_2_diag = torch.diag(q_i_2).unsqueeze(1)

        logits = torch.cat((q_k_diag, q_i_1_diag, q_i_2_diag), 1) / self.temperature # torch.Size([B, 3])

        labels1 = torch.ones((logits.shape[0],1)) 
        labels2 = torch.zeros((logits.shape[0],1))
        labels3 = torch.zeros((logits.shape[0],1))
        labels = torch.cat([labels1, labels2, labels3],dim=1).cuda()

        weight1 = torch.ones((logits.shape[0],1))
        weight2 = torch.ones((logits.shape[0],1))+2
        weight3 = torch.ones((logits.shape[0],1))+2
        weights = torch.cat([weight1, weight2, weight3],dim=1).cuda()

        loss = nn.BCEWithLogitsLoss(weight=weights)(logits, labels)

        return loss.mean()
    




# # ------------------------一个数据增强--------------------------------消融实验
# class simCLRLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(simCLRLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, adv_pc_logits, aug_pc1_logits):
#         # Normalize the features
#         aug_pc1_logits = F.normalize(aug_pc1_logits, p=2, dim=1)
#         # aug_pc2_logits = F.normalize(aug_pc2_logits, p=2, dim=1)
#         adv_pc_logits = F.normalize(adv_pc_logits, p=2, dim=1)

#         # 计算相似矩阵对角线上的相似度 经过F.normalize(p=2,dim=1)平方和开根号为1(余弦相似度)
#         # q_k = torch.mm(aug_pc1_logits, aug_pc2_logits.t()) 
#         q_i_1 = torch.mm(aug_pc1_logits, adv_pc_logits.t())
#         # q_i_2 = torch.mm(aug_pc2_logits, adv_pc_logits.t())

#         # q_k_diag = torch.diag(q_k).unsqueeze(1)
#         q_i_1_diag = torch.diag(q_i_1).unsqueeze(1)
#         # q_i_2_diag = torch.diag(q_i_2).unsqueeze(1)

#         # logits = torch.cat((q_k_diag, q_i_1_diag, q_i_2_diag), 1) / self.temperature # torch.Size([B, 3])
#         logits = q_i_1_diag / self.temperature
#         # logits = torch.tanh(logits)

#         # labels1 = torch.ones((logits.shape[0],1)) 
#         labels2 = torch.zeros((logits.shape[0],1)).cuda()
#         # labels3 = torch.zeros((logits.shape[0],1))
#         # labels = torch.cat([labels1, labels2, labels3],dim=1).cuda()

#         # weight1 = torch.ones((logits.shape[0],1))
#         # weight2 = torch.ones((logits.shape[0],1))+2
#         # weight3 = torch.ones((logits.shape[0],1))+2
#         # weights = torch.cat([weight1, weight2, weight3],dim=1).cuda()

#         loss = nn.BCEWithLogitsLoss()(logits, labels2)

#         return loss.mean()








# #-------------------------两个数据增强----------------------------------消融实验
# class simCLRLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(simCLRLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, adv_pc_logits, aug_pc1_logits, aug_pc2_logits):
#         # Normalize the features
#         aug_pc1_logits = F.normalize(aug_pc1_logits, p=2, dim=1)
#         aug_pc2_logits = F.normalize(aug_pc2_logits, p=2, dim=1)
#         adv_pc_logits = F.normalize(adv_pc_logits, p=2, dim=1)

#         # 计算相似矩阵对角线上的相似度 经过F.normalize(p=2,dim=1)平方和开根号为1(余弦相似度)
#         # q_k = torch.mm(aug_pc1_logits, aug_pc2_logits.t()) 
#         q_i_1 = torch.mm(aug_pc1_logits, adv_pc_logits.t())
#         q_i_2 = torch.mm(aug_pc2_logits, adv_pc_logits.t())

#         # q_k_diag = torch.diag(q_k).unsqueeze(1)
#         q_i_1_diag = torch.diag(q_i_1).unsqueeze(1)
#         q_i_2_diag = torch.diag(q_i_2).unsqueeze(1)

#         logits = torch.cat((q_i_1_diag, q_i_2_diag), 1) / self.temperature # torch.Size([B, 3])

#         # logits = torch.tanh(logits)

#         # labels1 = torch.ones((logits.shape[0],1)) 
#         labels2 = torch.zeros((logits.shape[0],1))
#         labels3 = torch.zeros((logits.shape[0],1))
#         labels = torch.cat([labels2, labels3],dim=1).cuda()

#         loss = nn.BCEWithLogitsLoss()(logits, labels)

#         return loss.mean()


# ------------------------三个数据增强--------------------------------消融实验
# class simCLRLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(simCLRLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, adv_pc_logits, aug_pc1_logits, aug_pc2_logits, aug_pc3_logits):
#         # Normalize the features
#         aug_pc1_logits = F.normalize(aug_pc1_logits, p=2, dim=1)
#         aug_pc2_logits = F.normalize(aug_pc2_logits, p=2, dim=1)
#         aug_pc3_logits = F.normalize(aug_pc3_logits, p=2, dim=1)
#         adv_pc_logits = F.normalize(adv_pc_logits, p=2, dim=1)

#         # 计算相似矩阵对角线上的相似度 经过F.normalize(p=2,dim=1)平方和开根号为1(余弦相似度)
#         q_k_1 = torch.mm(aug_pc1_logits, aug_pc2_logits.t())
#         q_k_2 = torch.mm(aug_pc1_logits, aug_pc3_logits.t())
#         q_k_3 = torch.mm(aug_pc2_logits, aug_pc3_logits.t())

#         q_i_1 = torch.mm(aug_pc1_logits, adv_pc_logits.t())
#         q_i_2 = torch.mm(aug_pc2_logits, adv_pc_logits.t())
#         q_i_3 = torch.mm(aug_pc3_logits, adv_pc_logits.t())

#         q_k_1_diag = torch.diag(q_k_1).unsqueeze(1)
#         q_k_2_diag = torch.diag(q_k_2).unsqueeze(1)
#         q_k_3_diag = torch.diag(q_k_3).unsqueeze(1)

#         q_i_1_diag = torch.diag(q_i_1).unsqueeze(1)
#         q_i_2_diag = torch.diag(q_i_2).unsqueeze(1)
#         q_i_3_diag = torch.diag(q_i_3).unsqueeze(1)

#         logits = torch.cat((q_k_1_diag, q_k_2_diag, q_k_3_diag, q_i_1_diag, q_i_2_diag, q_i_3_diag), 1) / self.temperature # torch.Size([B, 3])
#         # logits = q_i_1_diag / self.temperature
#         # logits = torch.tanh(logits)

#         labels1 = torch.ones((logits.shape[0],1)) 
#         labels2 = torch.zeros((logits.shape[0],1))
#         # labels3 = torch.zeros((logits.shape[0],1))
#         labels = torch.cat([labels1, labels1, labels1, labels2, labels2, labels2],dim=1).cuda()

#         weight1 = torch.ones((logits.shape[0],1))
#         weight2 = torch.ones((logits.shape[0],1))+2

#         weights = torch.cat([weight1, weight1, weight1, weight2, weight2, weight2],dim=1).cuda()

#         loss = nn.BCEWithLogitsLoss(weights)(logits, labels)

#         return loss.mean()





class simCLRLoss_fea(nn.Module):

    def __init__(self, temperature=0.07):
        super(simCLRLoss_fea, self).__init__()
        self.temperature = temperature
    
    def forward(self, ori_pc_fea, adv_pc_fea): # 只考虑负例，生成对抗样本
        # 对特征进行归一化
        ori_pc_fea = F.normalize(ori_pc_fea, p=2, dim=1)
        adv_pc_fea = F.normalize(adv_pc_fea, p=2, dim=1)

        # q_k = torch.mm(ori_pc_fea, ori_pc_fea.t()) / self.temperature
        q_i = torch.mm(ori_pc_fea, adv_pc_fea.t()) / self.temperature

        q_i_diag = torch.diag_embed(torch.diag(q_i_1))
        labels = q_i - q_i_diag

        loss = nn.CrossEntropyLoss()(logits, labels)

        
        return loss.sum()
    