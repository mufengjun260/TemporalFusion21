from collections import deque

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from lib.pspnet import PSPNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)


def to_aug_mat(input_mat):
    return torch.cat([input_mat, torch.ones((input_mat.shape[0], 1)).cuda()], dim=1).transpose(0, 1)


def merge_pc(cur_cloud, last_pose, init_pose):
    pred_pose = torch.mm(init_pose.cpu(), last_pose)
    pred_r = torch.as_tensor(quaternion_from_matrix(pred_pose[0:3, 0:3]).T, dtype=torch.float32).view(1, 4, 1).cuda()
    pred_t = pred_pose[0:3, 3].view(1, 3, 1).cuda()
    return cur_cloud[0].reshape(1, -1, 35), pred_r, pred_t


def getMaxRt(out_rx, out_cx, out_tx, points):
    bs, num_p, _ = out_cx.size()

    how_max, which_max = torch.max(out_cx, 1)

    pred_r = out_rx[0][which_max[0]][0] / (torch.norm(out_rx[0][which_max[0]][0], dim=0))
    pred_t = out_tx[0, which_max[0]].view(3) + points[0, which_max[0], 0:3].view(3)

    return pred_r.detach().cpu(), pred_t.detach().cpu()


class TemporalFeat(nn.Module):
    def __init__(self, num_points):
        super(TemporalFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(35, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 256, 1)
        # Channel Attention
        self.avg_pool_channel = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool_channel = torch.nn.AdaptiveMaxPool1d(1)
        self.fc1_avg = torch.nn.Conv1d(1024, 256, 1)
        self.fc2_avg = torch.nn.Conv1d(256, 1024, 1)
        self.fc1_max = torch.nn.Conv1d(1024, 256, 1)
        self.fc2_max = torch.nn.Conv1d(256, 1024, 1)

        self.avg_weight = torch.nn.Parameter(torch.as_tensor(1.))
        self.max_weight = torch.nn.Parameter(torch.as_tensor(1.))

        self.pr1 = torch.nn.Conv1d(4, 64, 1)
        self.pt1 = torch.nn.Conv1d(3, 64, 1)
        self.pr2 = torch.nn.Conv1d(64, 128, 1)
        self.pt2 = torch.nn.Conv1d(64, 128, 1)
        self.pr3 = torch.nn.Conv1d(128, 256, 1)
        self.pt3 = torch.nn.Conv1d(128, 256, 1)

    def forward(self, x, pred_r_from_last, pred_t_from_last):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_1 = F.relu(self.conv4(x))

        pr = F.relu(self.pr1(pred_r_from_last))
        pr = F.relu(self.pr2(pr))
        pr = F.relu(self.pr3(pr))

        pt = F.relu(self.pt1(pred_t_from_last))
        pt = F.relu(self.pt2(pt))
        pt = F.relu(self.pt3(pt))

        pr = pr.view(1, -1, 1).repeat(1, 1, x_1.shape[2])
        pt = pt.view(1, -1, 1).repeat(1, 1, x_1.shape[2])

        x_1_global = F.relu(self.conv5(self.avg_weight * self.avg_pool_channel(x_1) +
                                       self.max_weight * self.max_pool_channel(x_1)))
        x_1_global = x_1_global.view(-1, 256, 1).repeat(1, 1, x_1.shape[2])

        x_merged = torch.cat([x_1, x_1_global, pr, pt], 1)

        # ChannelAttention
        avg_pool_channel_out = self.fc2_avg(F.relu(self.fc1_avg(self.avg_pool_channel(x_merged))))
        max_pool_channel_out = self.fc2_max(F.relu(self.fc1_max(self.max_pool_channel(x_merged))))
        channel_attention_out = F.sigmoid(avg_pool_channel_out + max_pool_channel_out)

        x_att_merged = channel_attention_out * x_merged

        return x_att_merged


class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj, object_max):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        for p in self.parameters():
            p.requires_grad = False

        self.temporal_feat = TemporalFeat(num_points)

        self.conv1_r = torch.nn.Conv1d(1024, 512, 1)
        self.conv1_t = torch.nn.Conv1d(1024, 512, 1)
        self.conv1_c = torch.nn.Conv1d(1024, 512, 1)

        self.conv2_r = torch.nn.Conv1d(512, 256, 1)
        self.conv2_t = torch.nn.Conv1d(512, 256, 1)
        self.conv2_c = torch.nn.Conv1d(512, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj * 4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj * 3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj * 1, 1)  # confidence

        self.num_obj = num_obj

        self.last_R = {key: [] for key in range(0, object_max)}
        self.last_t = {key: [] for key in range(0, object_max)}

        self.last_R_total = {key: [] for key in range(0, object_max)}
        self.last_t_total = {key: [] for key in range(0, object_max)}
        self.last_c_total = {key: [] for key in range(0, object_max)}
        self.last_x_total = {key: [] for key in range(0, object_max)}

        for i in range(0, object_max):
            self.last_R[i] = None
            self.last_t[i] = None
            self.last_R_total[i] = None
            self.last_t_total[i] = None
            self.last_c_total[i] = None
            self.last_x_total[i] = None

    def forward(self, img, x, choose, obj, focal_length, principal_point, motion, is_train):
        if img.shape[1] == 0 and is_train:
            if self.last_R_total[int(obj)] is None:
                return None, None, None, None
            last_pose = torch.cat([torch.cat([self.last_R_total[int(obj)],
                                              (self.last_t_total[int(obj)] + self.last_x_total[int(obj)]).
                                             reshape(1000, 3, 1)], dim=2),
                                   torch.as_tensor([[0, 0, 0, 1]], dtype=torch.float32).repeat(1000, 1, 1).cuda()],
                                  dim=1)
            init_pose = motion[0].cuda().repeat(1000, 1, 1)
            transformed_pose = torch.bmm(init_pose, last_pose)
            self.last_R_total[int(obj)] = transformed_pose[:, 0:3, 0:3]
            self.last_t_total[int(obj)] = transformed_pose[:, 0:3, 3].reshape(1, 1000, 3) - self.last_x_total[int(obj)]
            out_rx = self.last_R_total[int(obj)]
            out_tx = self.last_t_total[int(obj)]
            out_cx = self.last_c_total[int(obj)]
            out_x = self.last_x_total[int(obj)]
            return out_rx, out_tx, out_cx, out_x

        out_img = self.cnn(img)
        bs, di, _, _ = out_img.size()
        choose_label = choose.repeat(1, di, 1)
        label_img = out_img.view(bs, di, -1)
        x_label = torch.gather(label_img, 2, choose_label).transpose(1, 2)

        x_six = torch.cat([x, x_label], 2)

        pred_r_from_last = torch.as_tensor([1., 0, 0, 0]).view(1, 4, 1).cuda()
        pred_t_from_last = torch.as_tensor([1., 0, 0]).view(1, 3, 1).cuda()

        if self.last_R[int(obj)] is not None:

            init_pose = motion[0].cuda()
            last_R_matrix = quaternion_matrix(self.last_R[int(obj)])[0:3, 0:3].T
            last_pose = torch.cat([torch.cat(
                [torch.as_tensor(last_R_matrix, dtype=torch.float32), self.last_t[int(obj)].view(3, 1)], dim=1),
                torch.as_tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0)

            x_six, pred_r_from_last, pred_t_from_last = merge_pc(x_six, last_pose, init_pose)

        temporal_x = self.temporal_feat(x_six.transpose(1, 2), pred_r_from_last, pred_t_from_last)

        rx = F.relu(self.conv1_r(temporal_x))
        tx = F.relu(self.conv1_t(temporal_x))
        cx = F.relu(self.conv1_c(temporal_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, -1)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, -1)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, -1)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        pred_r, pred_t = getMaxRt(out_rx, out_cx, out_tx, x_six)
        self.last_R[int(obj)] = pred_r
        self.last_t[int(obj)] = pred_t

        out_rx = self.Q2R(out_rx)
        out_x = x_six[:, :, 0:3]
        # save current state
        if self.last_R_total[int(obj)] is not None:

            last_pose = torch.cat([torch.cat([self.last_R_total[int(obj)],
                                              (self.last_t_total[int(obj)] + self.last_x_total[int(obj)]).
                                             reshape(1000, 3, 1)], dim=2),
                                   torch.as_tensor([[0, 0, 0, 1]], dtype=torch.float32).repeat(1000, 1, 1).cuda()],
                                  dim=1)
            init_pose = motion[0].cuda().repeat(1000, 1, 1)
            transformed_pose = torch.bmm(init_pose, last_pose)
            self.last_R_total[int(obj)] = transformed_pose[:, 0:3, 0:3]
            self.last_t_total[int(obj)] = transformed_pose[:, 0:3, 3].reshape(1, 1000, 3) - self.last_x_total[int(obj)]

            lrt = torch.cat([self.last_R_total[int(obj)], out_rx], dim=0)
            ltt = torch.cat([self.last_t_total[int(obj)], out_tx], dim=1)
            lct = torch.cat([self.last_c_total[int(obj)], out_cx], dim=1)
            lxt = torch.cat([self.last_x_total[int(obj)], x_six[:, :, 0:3]], dim=1)

            out_rx = torch.cat([self.last_R_total[int(obj)], out_rx], dim=0)
            out_tx = torch.cat([self.last_t_total[int(obj)], out_tx], dim=1)
            out_cx = torch.cat([self.last_c_total[int(obj)], out_cx], dim=1)
            out_x = torch.cat([self.last_x_total[int(obj)], x_six[:, :, 0:3]], dim=1)

            mask = np.zeros(lrt.shape[0], dtype=int)
            mask[:1000] = 1
            np.random.shuffle(mask)
            lrt = lrt[mask.nonzero(), :, :][0]
            ltt = ltt[0, mask.nonzero(), :]
            lct = lct[0, mask.nonzero(), :]
            lxt = lxt[0, mask.nonzero(), :]

            self.last_R_total[int(obj)] = lrt
            self.last_t_total[int(obj)] = ltt
            self.last_c_total[int(obj)] = lct
            self.last_x_total[int(obj)] = lxt
        else:
            self.last_R_total[int(obj)] = out_rx
            self.last_t_total[int(obj)] = out_tx
            self.last_c_total[int(obj)] = out_cx
            self.last_x_total[int(obj)] = x_six[:, :, 0:3]

        return out_rx, out_tx, out_cx, out_x

    def Q2R(self, pred_r):
        bs, num_p, _ = pred_r.size()
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
        return torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                          (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs,
                                                                                                                   num_p,
                                                                                                                   1), \
                          (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                                   num_p,
                                                                                                                   1), \
                          (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs,
                                                                                                                   num_p,
                                                                                                                   1), \
                          (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                          (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                                    num_p,
                                                                                                                    1), \
                          (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                                    num_p,
                                                                                                                    1), \
                          (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                                   num_p,
                                                                                                                   1), \
                          (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)),
                         dim=2).contiguous().view(bs * num_p, 3, 3).transpose(1, 2)

    def temporalClear(self, object_max):
        for i in range(0, object_max):
            self.last_R[i] = None
            self.last_t[i] = None
            self.last_R_total[i] = None
            self.last_t_total[i] = None
            self.last_c_total[i] = None
            self.last_x_total[i] = None


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x


class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj * 4)  # quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj * 3)  # translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
