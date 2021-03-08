import pickle

import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import cv2


class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, seg_type, with_non_input):
        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_temporal.txt'
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_temporal.txt'

        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.seg_type = seg_type
        self.with_non_input = with_non_input

        self.list = []
        self.real = []
        self.syn = []
        self.list_code = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('datasets/ycb/dataset_config/classes.txt')
        class_id = 1
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()

            class_id += 1

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.front_num = 2

        print(len(self.list))

    def __getitem__(self, index):
        # print(index)
        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        if self.seg_type == "seg":
            label = np.array(Image.open('{0}/{1}-seg.png'.format(self.root, self.list[index])))
        elif self.seg_type == "masked":
            label = np.array(Image.open('{0}/{1}-seg-masked-square.png'.format(self.root, self.list[index])))
        else:
            label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        motion = np.array([[1., 0, 0, 0], [0, 1, 0, 0, ], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        if os.path.exists('{0}/{1}-odom-hybrid.pickle'.format(self.root, self.list[index])):
            motion = pickle.load(open('{0}/{1}-odom-hybrid.pickle'.format(self.root, self.list[index]), 'rb'))

        focal_length = meta.get("intrinsic_matrix")[0][0]
        principal_point = [meta.get("intrinsic_matrix")[0][2], meta.get("intrinsic_matrix")[1][2]]

        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                    continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        # init lists
        list_clouds = []
        list_chooses = []
        list_image_masked = []
        list_target = []
        list_points = []
        list_obj = []
        list_filename = []
        list_full_img = []
        list_focal_length = []
        list_principal_point = []
        list_motion = []
        for idx in range(len(obj)):
            local_img = img
            full_img = img
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) < self.minimum_num_pt:
                if self.with_non_input:
                    print('dropped frame: {0} with {1} pixel'.format(self.list[index], len(mask.nonzero()[0])))
                    full_img = np.transpose(np.array(full_img)[:, :, :3], (2, 0, 1))

                    dellist = [j for j in range(0, len(self.cld[obj[idx]]))]

                    dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
                    model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

                    target_r = meta['poses'][:, :, idx][:, 0:3]
                    target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
                    add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

                    target = np.dot(model_points, target_r.T)
                    # print('target r for ', idx, ' : ', target_r.T)
                    if self.add_noise:
                        target = np.add(target, target_t + add_t)
                    else:
                        # print('target t for ', idx, ' : ', target_t)
                        target = np.add(target, target_t)

                    filename = self.list[index]

                    list_clouds.append(torch.tensor([]))
                    list_chooses.append(torch.tensor([]))
                    list_image_masked.append(torch.tensor([]))
                    list_target.append(torch.from_numpy(target.astype(np.float32)))
                    list_points.append(torch.from_numpy(model_points.astype(np.float32)))
                    list_obj.append(torch.LongTensor([int(obj[idx]) - 1]))
                    list_filename.append(filename)
                    list_full_img.append(full_img)
                    list_focal_length.append(focal_length)
                    list_principal_point.append(principal_point)
                    list_motion.append(motion)
                continue

            if self.add_noise:
                local_img = self.trancolor(local_img)

            rmin, rmax, cmin, cmax = get_bbox(mask_label)
            local_img = np.transpose(np.array(local_img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            full_img = np.transpose(np.array(full_img)[:, :, :3], (2, 0, 1))

            if self.list[index][:8] == 'data_syn':
                seed = random.choice(self.real)
                back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
                img_masked = back * mask_back[rmin:rmax, cmin:cmax] + local_img
            else:
                img_masked = local_img

            if self.add_noise and add_front:
                img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(
                    mask_front[rmin:rmax, cmin:cmax])

            if self.list[index][:8] == 'data_syn':
                img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

            target_r = meta['poses'][:, :, idx][:, 0:3]
            target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
            add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) > self.num_pt:

                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cam_scale = meta['factor_depth'][0][0]
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            if self.add_noise:
                cloud = np.add(cloud, add_t)

            dellist = [j for j in range(0, len(self.cld[obj[idx]]))]

            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
            model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

            target = np.dot(model_points, target_r.T)
            if self.add_noise:
                target = np.add(target, target_t + add_t)
            else:
                target = np.add(target, target_t)
            filename = self.list[index]
            list_clouds.append(torch.from_numpy(cloud.astype(np.float32)))
            list_chooses.append(torch.LongTensor(choose.astype(np.int32)))
            list_image_masked.append(self.norm(torch.from_numpy(img_masked.astype(np.float32))))
            list_target.append(torch.from_numpy(target.astype(np.float32)))
            list_points.append(torch.from_numpy(model_points.astype(np.float32)))
            list_obj.append(torch.LongTensor([int(obj[idx]) - 1]))
            list_filename.append(filename)
            list_full_img.append(full_img)
            list_focal_length.append(focal_length)
            list_principal_point.append(principal_point)
            list_motion.append(motion)
        return list_clouds, \
               list_chooses, \
               list_image_masked, \
               list_target, \
               list_points, \
               list_obj, \
               list_filename, \
               list_full_img, \
               list_focal_length, \
               list_principal_point, \
               list_motion

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
