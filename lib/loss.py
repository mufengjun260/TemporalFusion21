from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.knn.__init__ import KNearestNeighbor

def loss_calculation(pred_r, pred_t, pred_c, dis_vector_last, target, model_points, idx, points,
                     w, refine, num_point_mesh, sym_list, stable_alpha):
    knn = KNearestNeighbor(1)
    bs, num_p, _ = pred_c.size()
    base = pred_r
    ori_base = pred_r.transpose(1, 2)
    base = base.contiguous()

    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh,
                                                                                           3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)

    pred = torch.add(torch.bmm(model_points, base), points + pred_t)

    if not refine:
        if idx[0].item() in sym_list:
            target = target[0].transpose(1, 0).contiguous().view(3, -1)
            pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
            inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
            target = torch.index_select(target, 1, inds.view(-1).detach() - 1)
            target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)

    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    dis = dis.view(bs, num_p)
    dis_vector = torch.mean((pred - target)[which_max[0]], dim=0)
    if dis_vector_last is None: dis_vector_last = dis_vector
    loss_stable = loss + stable_alpha * torch.norm(dis_vector - dis_vector_last,
                                                   dim=0)

    t = ori_t[which_max[0]] + points[which_max[0]]
    points = points.view(1, bs * num_p, 3)
    ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3)

    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)

    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    del knn
    return loss_stable, dis[0][which_max[0]], new_points.detach(), new_target.detach(), dis_vector


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, dis_vector_last, target, model_points, idx, points,
                w, refine, stable_alpha):
        return loss_calculation(pred_r, pred_t, pred_c, dis_vector_last, target, model_points,
                                idx, points, w, refine, self.num_pt_mesh,
                                self.sym_list, stable_alpha)
