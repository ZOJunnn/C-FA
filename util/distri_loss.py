import torch
import torch.nn as nn
import torch.nn.functional as F

class DistriLoss(nn.Module):
    def __init__(self):
        super(DistriLoss, self).__init__()

    def forward(self, adv_data, aug_pc1_data, aug_pc2_data):
        aug_sim1 = (aug_pc1_data* aug_pc2_data).sum(dim=1)
        fenmu = torch.sqrt((aug_pc1_data**2).sum(dim = 1)) * torch.sqrt((aug_pc2_data**2).sum(dim = 1))
        similar = torch.abs(aug_sim1 / fenmu)

        adv_1 = (aug_pc1_data * adv_data).sum(dim=1)
        fenmu_adv_1 = torch.sqrt((aug_pc1_data ** 2).sum(dim=1)) * torch.sqrt((adv_data ** 2).sum(dim=1))
        similar_adv_1 = torch.abs(adv_1 / fenmu_adv_1)

        adv_2 = (aug_pc2_data * adv_data).sum(dim=1)
        fenmu_adv_2 = torch.sqrt((aug_pc2_data ** 2).sum(dim=1)) * torch.sqrt((adv_data ** 2).sum(dim=1))
        similar_adv_2 = torch.abs(adv_2 / fenmu_adv_2)

        # loss = 1/torch.tanh((torch.abs(similar_adv_1)- similar)) +  1/torch.tanh((torch.abs(similar_adv_2)- similar))
        loss = 1/(torch.sigmoid(nn.MSELoss()(similar, similar_adv_1))+1/torch.sigmoid(nn.MSELoss()(similar, similar_adv_2)))
        return loss.mean()




# if __name__ == "__main__":
#     loss = DistriLoss()
#     a = torch.tensor([
#         [[1,2],[2,3]],
#         [[2,2],[1,3]]

#     ])

#     b = torch.tensor([
#         [[3,2],[1,3]],
#         [[2,2],[2,3]]

#     ])
#     c = torch.tensor([
#         [[2,2],[3,3]],
#         [[2,2],[1,3]]

#     ])
#     loss.forward(a,b,c)