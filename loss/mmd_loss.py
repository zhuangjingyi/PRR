from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as  F

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, margin=0.0,**kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.margin = margin
        # print('COMMON MMDLoss : ')
        # print('margin: ',self.margin)


    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        # source = F.normalize(source, p=2, dim=1)
        # target = F.normalize(target, p=2, dim=1)
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            # TT = F.relu(XX + YY - XY - YX-self.margin)



            TT = XX + YY - XY - YX
            # print(TT)
            # TT = F.softplus(TT/self.margin)
            #
            # idx = TT>self.margin
            # loss = torch.mean(TT[idx])
            # if idx.sum()>0:
            #     loss = torch.mean(TT[idx])
            # else:
            #     loss = torch.zeros(1).cuda()
            # print(TT.shape)
            loss = torch.mean(TT)


            return loss



class ZJYLoss(nn.Module):
    def __init__(self, margin=0.0,**kwargs):
        super(ZJYLoss, self).__init__()
        self.margin = margin

    def forward(self, qi, qd):
        zero = torch.zeros_like(qi-qd)
        loss = torch.max(qi-qd+self.margin, zero)
        # print(loss.size())
        return loss


class FDALoss(nn.Module):
    def __init__(self, margin=0.0,**kwargs):
        super(FDALoss, self).__init__()
        self.margin = margin

    def forward(self, q):

        loss = F.softplus(-q)
        # print(loss.size())
        return loss