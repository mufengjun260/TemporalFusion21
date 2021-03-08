import copy

from PIL import Image
from PIL import ImageDraw

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from datasets.ycb.dataset_with_non_input import PoseDataset as PoseDataset_ycb
from lib.network import PoseNet
from lib.loss import Loss
from lib.utils import setup_logger
import torch.multiprocessing
import scipy.io as scio
from lib.knn.__init__ import KNearestNeighbor

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default='',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--output_dir', type=str, default='', help='output dir')
parser.add_argument('--seg_type', type=str, default="ori", help='output dir')
parser.add_argument('--object_max', type=int, default=21, help='length of classes.txt')

opt = parser.parse_args()
knn = KNearestNeighbor(1)
is_debug = False


def get_target(root, filename, obj):
    meta = scio.loadmat('{0}/{1}-meta.mat'.format(root, filename[0]))
    target_r = None
    target_t = None
    if len(np.where(meta['cls_indexes'].flatten() == int(obj) + 1)[0]) == 1:
        idx = int(np.where(meta['cls_indexes'].flatten() == int(obj) + 1)[0])
        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
    return torch.tensor(target_r).cuda(), torch.tensor(target_t).cuda()


def output_transformed_image(IMAGE_OUTPUT_PATH, output_image, cloud, focal_length, principal_point, obj):
    R = 2
    NUM_P = 500

    camera_k = np.array([focal_length[0].numpy(), 0.0, principal_point[0][0].numpy(), 0.0, focal_length[0].numpy(),
                         principal_point[1][0].numpy(), 0.0, 0.0, 1.0]).reshape(3, 3)
    img = output_image
    cloud = cloud.detach().cpu().numpy()

    proj_points = camera_k.dot(cloud.T)
    proj_points = proj_points / proj_points[2, :]
    proj_points = proj_points.T
    proj_points = np.delete(proj_points, 2, axis=1)

    if len(proj_points) > NUM_P:
        c_mask = np.zeros(len(proj_points), dtype=int)
        c_mask[:NUM_P] = 1
        np.random.shuffle(c_mask)
        proj_points = proj_points[c_mask.nonzero()]

    np.random.seed(obj)
    r = int(np.random.randint(0, 255))
    np.random.seed(obj * 144)
    g = int(np.random.randint(0, 255))
    np.random.seed(obj * 262)
    b = int(np.random.randint(0, 255))
    draw = ImageDraw.Draw(img)
    for i in range(len(proj_points)):
        draw.ellipse((proj_points[i][0] - R, proj_points[i][1] - R, proj_points[i][0] + R, proj_points[i][1] + R),
                     fill=(r, g, b))

    draw.text((10, 10), IMAGE_OUTPUT_PATH)

    return img
    # print(filename)


def main():
    class_id = 0
    class_file = open('datasets/ycb/dataset_config/classes.txt')
    cld = {}
    while 1:
        class_input = class_file.readline()
        if not class_input:
            break

        input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input[:-1]))
        cld[class_id] = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        cld[class_id] = np.array(cld[class_id])
        input_file.close()

        class_id += 1

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    symmetry_obj_idx = [12, 15, 18, 19, 20]

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = 'trained_models/ycb/' + opt.output_dir  # folder to save trained models
        opt.test_output = 'experiments/output/ycb/' + opt.output_dir
        if not os.path.exists(opt.test_output): os.makedirs(opt.test_output, exist_ok=True)

        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects, object_max=opt.object_max)
    estimator.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

        opt.refine_start = False
        opt.decay_start = False

    dataset = PoseDataset_ycb('train', opt.num_points, False, opt.dataset_root, opt.noise_trans,
                              opt.seg_type, True)
    test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.seg_type, True)

    testdataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                 num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)

    logger = setup_logger('final_eval_tf_with_seg_square',
                          os.path.join(opt.test_output, 'final_eval_tf_with_seg_square.txt'))

    object_max = opt.object_max
    total_test_dis = {key: [] for key in range(0, object_max)}
    total_test_count = {key: [] for key in range(0, object_max)}
    dir_test_dis = {key: [] for key in range(0, object_max)}
    dir_test_count = {key: [] for key in range(0, object_max)}

    # for add
    total_unseen_objects = {key: [] for key in range(0, object_max)}
    total_object_without_pose = {key: [] for key in range(0, object_max)}
    dir_add_count = {key: [] for key in range(0, object_max)}
    dir_add_count_unseen = {key: [] for key in range(0, object_max)}
    dir_add_02_count_unseen = {key: [] for key in range(0, object_max)}
    dir_add_pure_count = {key: [] for key in range(0, object_max)}
    dir_add_s_count = {key: [] for key in range(0, object_max)}
    dir_add_02_count = {key: [] for key in range(0, object_max)}
    dir_add_pure_02_count = {key: [] for key in range(0, object_max)}
    dir_add_s_02_count = {key: [] for key in range(0, object_max)}

    total_add_count = {key: [] for key in range(0, object_max)}
    total_add_count_unseen = {key: [] for key in range(0, object_max)}
    total_add_02_count_unseen = {key: [] for key in range(0, object_max)}
    total_add_pure_count = {key: [] for key in range(0, object_max)}
    total_add_s_count = {key: [] for key in range(0, object_max)}
    total_add_02_count = {key: [] for key in range(0, object_max)}
    total_add_pure_02_count = {key: [] for key in range(0, object_max)}
    total_add_s_02_count = {key: [] for key in range(0, object_max)}

    dir_dbd_count = {key: [] for key in range(0, object_max)}
    dir_drr_count = {key: [] for key in range(0, object_max)}
    dir_ada_count = {key: [] for key in range(0, object_max)}
    dir_distance_1_count = {key: [] for key in range(0, object_max)}

    total_dbd_count = {key: [] for key in range(0, object_max)}
    total_drr_count = {key: [] for key in range(0, object_max)}
    total_ada_count = {key: [] for key in range(0, object_max)}
    total_distance_1_count = {key: [] for key in range(0, object_max)}

    last_dis = {key: [] for key in range(0, object_max)}
    for i in range(object_max):
        total_unseen_objects[i] = 0
        total_object_without_pose[i] = 0

        total_test_dis[i] = 0.
        total_test_count[i] = 0
        dir_test_dis[i] = 0.
        dir_test_count[i] = 0
        # for add
        dir_add_count[i] = 0
        dir_add_count_unseen[i] = 0
        dir_add_02_count_unseen[i] = 0
        dir_add_pure_count[i] = 0
        dir_add_s_count[i] = 0
        dir_add_02_count[i] = 0
        total_add_count[i] = 0
        total_add_count_unseen[i] = 0
        total_add_02_count_unseen[i] = 0
        total_add_pure_count[i] = 0
        total_add_s_count[i] = 0
        total_add_02_count[i] = 0
        dir_add_pure_02_count[i] = 0
        dir_add_s_02_count[i] = 0
        total_add_pure_02_count[i] = 0
        total_add_s_02_count[i] = 0

        #   for stable
        dir_dbd_count[i] = 0.
        dir_drr_count[i] = 0
        dir_ada_count[i] = 0.
        dir_distance_1_count[i] = 0.

        total_dbd_count[i] = 0.
        total_drr_count[i] = 0
        total_ada_count[i] = 0.
        total_distance_1_count[i] = 0.
        last_dis[i] = None

    st_time = time.time()
    isFirstInitLastDatafolder = True
    estimator.eval()
    with torch.no_grad():
        for j, data in enumerate(testdataloader, 0):
            if opt.dataset == 'ycb':
                list_points, list_choose, list_img, list_target, list_model_points, list_idx, list_filename, \
                list_full_img, list_focal_length, list_principal_point, list_motion = data
            output_image = Image.open('{0}/{1}-color-masked-square.png'.format(opt.dataset_root, list_filename[0][0]))
            OUTPUT_IMAGE_PATH = '{0}/{1}-color-seg-square-output-tf.png'.format(opt.dataset_root, list_filename[0][0])
            for list_index in range(len(list_points)):
                points, choose, img, target, model_points, idx, filename, full_img, focal_length, principal_point, motion \
                    = list_points[list_index], list_choose[list_index], list_img[list_index], \
                      list_target[list_index], list_model_points[list_index], list_idx[list_index], \
                      list_filename[list_index], list_full_img[list_index], list_focal_length[list_index], \
                      list_principal_point[list_index], list_motion[list_index]

                # Temporal Clean when Changing datafolder
                datafolder = filename[0].split('/')[1]
                filehead = filename[0].split('/')[2]
                if isFirstInitLastDatafolder:
                    lastdatafolder = datafolder
                    isFirstInitLastDatafolder = False
                if datafolder != lastdatafolder:
                    logger.info('changing folder from {0} to {1}'.format(lastdatafolder, datafolder))
                    estimator.temporalClear(opt.object_max)
                    # handle dir output
                    for i in range(0, object_max):
                        if dir_test_count[i] != 0:
                            logger.info('Dir {0} Object {1} dis:{2} with {3} samples'.format(
                                lastdatafolder, i, dir_test_dis[i] / dir_test_count[i], dir_test_count[i]))
                            if dir_add_count[i] != 0:
                                logger.info('Dir {0} Object {1} add:{2} with 0.02: {3}'.format(
                                    lastdatafolder, i, dir_add_count[i] / dir_test_count[i],
                                                       dir_add_02_count[i] / dir_add_count[i]))
                            else:
                                logger.info('Dir {0} Object {1} add:{2} with 0.02: {3}'.format(
                                    lastdatafolder, i, dir_add_count[i] / dir_test_count[i], 0))
                            if dir_add_pure_count[i] != -0:
                                logger.info('Dir {0} Object {1} add_pure:{2} with 0.02: {3}'.format(
                                    lastdatafolder, i, dir_add_pure_count[i] / dir_test_count[i],
                                                       dir_add_pure_02_count[i] / dir_add_pure_count[i]))
                            else:
                                logger.info('Dir {0} Object {1} add_pure:{2} with 0.02: {3}'.format(
                                    lastdatafolder, i, dir_add_pure_count[i] / dir_test_count[i], 0))
                            if dir_add_s_count[i] != 0:
                                logger.info('Dir {0} Object {1} add_s:{2} with 0.02: {3}'.format(
                                    lastdatafolder, i, dir_add_s_count[i] / dir_test_count[i],
                                                       dir_add_s_02_count[i] / dir_add_s_count[i]))
                            else:
                                logger.info('Dir {0} Object {1} add_s:{2} with 0.02: {3}'.format(
                                    lastdatafolder, i, dir_add_s_count[i] / dir_test_count[i], 0))
                            logger.info('Dir {0} Object {1} dbd:{2}'.format(
                                lastdatafolder, i, dir_dbd_count[i] / dir_test_count[i]))
                            logger.info('Dir {0} Object {1} drr:{2}'.format(
                                lastdatafolder, i, dir_drr_count[i] / dir_test_count[i]))
                            logger.info('Dir {0} Object {1} ada:{2}'.format(
                                lastdatafolder, i, dir_ada_count[i] / dir_test_count[i]))
                            logger.info('Dir {0} Object {1} distance_1:{2}'.format(
                                lastdatafolder, i, dir_distance_1_count[i] / dir_test_count[i]))

                    dir_dbd = 0.
                    dir_drr = 0.
                    dir_ada = 0.
                    dir_distance_1 = 0.
                    dir_dis = 0.
                    dir_add = 0
                    dir_add_s = 0
                    dir_add_pure = 0
                    dir_add_02 = 0
                    dir_add_s_02 = 0
                    dir_add_pure_02 = 0
                    dir_count = 0

                    for i in range(object_max):
                        if total_test_count[i] != 0:
                            dir_count += dir_test_count[i]
                            dir_dis += dir_test_dis[i]
                            dir_add += dir_add_count[i]
                            dir_add_pure += dir_add_pure_count[i]
                            dir_add_s += dir_add_s_count[i]
                            dir_add_02 += dir_add_02_count[i]
                            dir_add_pure_02 += dir_add_pure_02_count[i]
                            dir_add_s_02 += dir_add_s_02_count[i]
                            dir_dbd += dir_dbd_count[i]
                            dir_drr += dir_drr_count[i]
                            dir_ada += dir_ada_count[i]
                            dir_distance_1 += dir_distance_1_count[i]

                            dir_test_dis[i] = 0
                            dir_test_count[i] = 0
                            dir_add_count[i] = 0
                            dir_add_pure_count[i] = 0
                            dir_add_s_count[i] = 0
                            dir_add_02_count[i] = 0
                            dir_add_pure_02_count[i] = 0
                            dir_add_s_02_count[i] = 0
                            dir_dbd_count[i] = 0
                            dir_drr_count[i] = 0
                            dir_ada_count[i] = 0
                            dir_distance_1_count[i] = 0
                            last_dis[i] = None

                    logger.info('Dir {0} \'s total dis:{1} with {2} samples'.format(
                        lastdatafolder, dir_dis / dir_count, dir_count))
                    logger.info('Dir {0} \'s total add:{1} with 0.02: {2}'.format(
                        lastdatafolder, dir_add / dir_count, dir_add_02 / dir_add))
                    logger.info('Dir {0} \'s total add_s:{1} with 0.02: {2}'.format(
                        lastdatafolder, dir_add_s / dir_count, dir_add_s_02 / dir_add_s))
                    logger.info('Dir {0} \'s total add_pure:{1} with 0.02: {2}'.format(
                        lastdatafolder, dir_add_pure / dir_count, dir_add_pure_02 / dir_add_pure))
                    logger.info('Dir {0} \'s total dbd:{1}'.format(lastdatafolder, dir_dbd / dir_count))
                    logger.info('Dir {0} \'s total drr:{1}'.format(lastdatafolder, dir_drr / dir_count))
                    logger.info('Dir {0} \'s total ada:{1}'.format(lastdatafolder, dir_ada / dir_count))
                    logger.info('Dir {0} \'s total distance_1:{1}'.format(lastdatafolder, dir_distance_1 / dir_count))

                    # end of handle dir output

                lastdatafolder = datafolder

                points, choose, img, target, model_points, idx = points.cuda(), \
                                                                 choose.cuda(), \
                                                                 img.cuda(), \
                                                                 target.cuda(), \
                                                                 model_points.cuda(), \
                                                                 idx.cuda()
                cloud_path = "experiments/clouds/ycb/{0}/{1}/{2}/{3}_{4}".format(opt.output_dir, 1,
                                                                                 datafolder, filehead,
                                                                                 int(idx))  # folder to save logs

                pred_r, pred_t, pred_c, x_return = estimator(img, points, choose, idx, focal_length,
                                                             principal_point, motion, cloud_path)

                # count for unseen object
                if pred_r is None:
                    last_dis[int(idx)] = None
                    total_unseen_objects[int(idx)] += 1
                    total_object_without_pose[int(idx)] += 1
                    continue

                pred_r_ori = copy.deepcopy(pred_r)
                pred_t_ori = copy.deepcopy(pred_t)
                pred_c_ori = copy.deepcopy(pred_c)
                x_return_ori = copy.deepcopy(x_return)

                gt_r, gt_t = get_target(opt.dataset_root, filename, idx)
                if gt_r is None: print('gtr is None')
                is_sym = int(idx) in symmetry_obj_idx
                dis, dis_vector, pred_cloud = calDistance(pred_r_ori, pred_t_ori, pred_c_ori, x_return_ori, gt_r, gt_t,
                                                          cld[int(idx)], is_sym)
                dis_s, dis_vector_s, _ = calDistance(pred_r_ori, pred_t_ori, pred_c_ori, x_return_ori, gt_r, gt_t,
                                                     cld[int(idx)], True)
                dis_pure, dis_vector_pure, _ = calDistance(pred_r_ori, pred_t_ori, pred_c_ori, x_return_ori, gt_r, gt_t,
                                                           cld[int(idx)],
                                                           False)

                if last_dis[int(idx)] is not None:
                    dir_dbd_count[int(idx)] += torch.norm(dis_vector - last_dis[int(idx)])
                    total_dbd_count[int(idx)] += torch.norm(dis_vector - last_dis[int(idx)])
                    dir_distance_1_count[int(idx)] += torch.norm(
                        (dis_vector / torch.norm(dis_vector)) - (last_dis[int(idx)] / torch.norm(last_dis[int(idx)])))
                    total_distance_1_count[int(idx)] += torch.norm(
                        (dis_vector / torch.norm(dis_vector)) - (last_dis[int(idx)] / torch.norm(last_dis[int(idx)])))
                    if torch.dot(last_dis[int(idx)], dis_vector) < 0:
                        dir_drr_count[int(idx)] += 1
                        total_drr_count[int(idx)] += 1
                    dir_ada_count[int(idx)] += torch.acos(
                        (torch.dot(last_dis[int(idx)], dis_vector)) / (
                                torch.norm(last_dis[int(idx)]) * torch.norm(dis_vector)))
                    total_ada_count[int(idx)] += torch.acos(
                        (torch.dot(last_dis[int(idx)], dis_vector)) / (
                                torch.norm(last_dis[int(idx)]) * torch.norm(dis_vector)))

                last_dis[int(idx)] = dis_vector

                # calc adds
                if img.shape[1] != 0:
                    dir_test_dis[int(idx)] += dis.item()

                    total_test_dis[int(idx)] += dis.item()
                    dir_test_count[int(idx)] += 1
                    total_test_count[int(idx)] += 1

                    if dis < 0.1:
                        dir_add_count[int(idx)] += 1
                        total_add_count[int(idx)] += 1
                    if dis < 0.02:
                        dir_add_02_count[int(idx)] += 1
                        total_add_02_count[int(idx)] += 1
                    if dis_s < 0.1:
                        dir_add_s_count[int(idx)] += 1
                        total_add_s_count[int(idx)] += 1
                    if dis_s < 0.02:
                        dir_add_s_02_count[int(idx)] += 1
                        total_add_s_02_count[int(idx)] += 1
                    if dis_pure < 0.1:
                        dir_add_pure_count[int(idx)] += 1
                        total_add_pure_count[int(idx)] += 1
                    if dis_pure < 0.02:
                        dir_add_pure_02_count[int(idx)] += 1
                        total_add_pure_02_count[int(idx)] += 1
                else:
                    last_dis[int(idx)] = None
                    if dis < 0.1:
                        dir_add_count_unseen[int(idx)] += 1
                        total_add_count_unseen[int(idx)] += 1
                        total_unseen_objects[int(idx)] += 1
                    if dis < 0.02:
                        dir_add_02_count_unseen[int(idx)] += 1
                        total_add_02_count_unseen[int(idx)] += 1
                        total_unseen_objects[int(idx)] += 1

                output_image = output_transformed_image(OUTPUT_IMAGE_PATH, output_image, pred_cloud, focal_length,
                                                        principal_point, int(idx))
                logger.info('Test time {0} Test Frame {1} {2} dis:{3}'.format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)),
                    filename,
                    idx.item(), dis))

            output_image.save(OUTPUT_IMAGE_PATH)

        # handle dir output
        for i in range(0, object_max):
            if dir_test_count[i] != 0:
                logger.info('Dir {0} Object {1} dis:{2} with {3} samples'.format(
                    lastdatafolder, i, dir_test_dis[i] / dir_test_count[i], dir_test_count[i]))
                if dir_add_count[i] != 0:
                    logger.info('Dir {0} Object {1} add:{2} with 0.02: {3}'.format(
                        lastdatafolder, i, dir_add_count[i] / dir_test_count[i],
                                           dir_add_02_count[i] / dir_add_count[i]))
                else:
                    logger.info('Dir {0} Object {1} add:{2} with 0.02: {3}'.format(
                        lastdatafolder, i, dir_add_count[i] / dir_test_count[i], 0))
                if dir_add_pure_count[i] != -0:
                    logger.info('Dir {0} Object {1} add_pure:{2} with 0.02: {3}'.format(
                        lastdatafolder, i, dir_add_pure_count[i] / dir_test_count[i],
                                           dir_add_pure_02_count[i] / dir_add_pure_count[i]))
                else:
                    logger.info('Dir {0} Object {1} add_pure:{2} with 0.02: {3}'.format(
                        lastdatafolder, i, dir_add_pure_count[i] / dir_test_count[i], 0))
                if dir_add_s_count[i] != 0:
                    logger.info('Dir {0} Object {1} add_s:{2} with 0.02: {3}'.format(
                        lastdatafolder, i, dir_add_s_count[i] / dir_test_count[i],
                                           dir_add_s_02_count[i] / dir_add_s_count[i]))
                else:
                    logger.info('Dir {0} Object {1} add_s:{2} with 0.02: {3}'.format(
                        lastdatafolder, i, dir_add_s_count[i] / dir_test_count[i], 0))
                logger.info('Dir {0} Object {1} dbd:{2}'.format(
                    lastdatafolder, i, dir_dbd_count[i] / dir_test_count[i]))
                logger.info('Dir {0} Object {1} drr:{2}'.format(
                    lastdatafolder, i, dir_drr_count[i] / dir_test_count[i]))
                logger.info('Dir {0} Object {1} ada:{2}'.format(
                    lastdatafolder, i, dir_ada_count[i] / dir_test_count[i]))
                logger.info('Dir {0} Object {1} distance_1:{2}'.format(
                    lastdatafolder, i, dir_distance_1_count[i] / dir_test_count[i]))

        dir_dbd = 0.
        dir_drr = 0.
        dir_ada = 0.
        dir_distance_1 = 0.
        dir_dis = 0.
        dir_add = 0
        dir_add_s = 0
        dir_add_pure = 0
        dir_add_02 = 0
        dir_add_s_02 = 0
        dir_add_pure_02 = 0
        dir_count = 0

        for i in range(object_max):
            if total_test_count[i] != 0:
                dir_count += dir_test_count[i]
                dir_dis += dir_test_dis[i]
                dir_add += dir_add_count[i]
                dir_add_pure += dir_add_pure_count[i]
                dir_add_s += dir_add_s_count[i]
                dir_add_02 += dir_add_02_count[i]
                dir_add_pure_02 += dir_add_pure_02_count[i]
                dir_add_s_02 += dir_add_s_02_count[i]
                dir_dbd += dir_dbd_count[i]
                dir_drr += dir_drr_count[i]
                dir_ada += dir_ada_count[i]
                dir_distance_1 += dir_distance_1_count[i]

                dir_test_dis[i] = 0
                dir_test_count[i] = 0
                dir_add_count[i] = 0
                dir_add_pure_count[i] = 0
                dir_add_s_count[i] = 0
                dir_add_02_count[i] = 0
                dir_add_pure_02_count[i] = 0
                dir_add_s_02_count[i] = 0
                dir_dbd_count[i] = 0
                dir_drr_count[i] = 0
                dir_ada_count[i] = 0
                dir_distance_1_count[i] = 0

        logger.info('Dir {0} \'s total dis:{1} with {2} samples'.format(
            lastdatafolder, dir_dis / dir_count, dir_count))
        logger.info('Dir {0} \'s total add:{1} with 0.02: {2}'.format(
            lastdatafolder, dir_add / dir_count, dir_add_02 / dir_add))
        logger.info('Dir {0} \'s total add_s:{1} with 0.02: {2}'.format(
            lastdatafolder, dir_add_s / dir_count, dir_add_s_02 / dir_add_s))
        logger.info('Dir {0} \'s total add_pure:{1} with 0.02: {2}'.format(
            lastdatafolder, dir_add_pure / dir_count, dir_add_pure_02 / dir_add_pure))
        logger.info('Dir {0} \'s total dbd:{1}'.format(lastdatafolder, dir_dbd / dir_count))
        logger.info('Dir {0} \'s total drr:{1}'.format(lastdatafolder, dir_drr / dir_count))
        logger.info('Dir {0} \'s total ada:{1}'.format(lastdatafolder, dir_ada / dir_count))
        logger.info('Dir {0} \'s total distance_1:{1}'.format(lastdatafolder, dir_distance_1 / dir_count))

        # end of handle dir output

        # handle global output
        total_unseen_count = 0
        total_without_pose_count = 0
        total_add_count_unseen_count = 0
        total_add_02_count_unseen_count = 0
        total_drr = 0.
        total_dbd = 0.
        total_ada = 0.
        total_distance_1 = 0.
        total_dis = 0.
        total_add = 0
        total_add_s = 0
        total_add_pure = 0
        total_add_02 = 0
        total_add_s_02 = 0
        total_add_pure_02 = 0
        total_count = 0
        for i in range(object_max):
            if total_test_count[i] != 0:
                logger.info('Total: Object {0} dis:{1} with {2} samples'.format(
                    i, total_test_dis[i] / total_test_count[i], total_test_count[i]))
                logger.info('Total: Object {0} add:{1} with 0.02: {2}'.format(
                    i, total_add_count[i] / total_test_count[i], total_add_02_count[i] / total_add_count[i]))
                logger.info('Total: Object {0} drr:{1}'.format(
                    i, total_drr_count[i] / total_test_count[i]))
                logger.info('Total: Object {0} ada:{1}'.format(
                    i, total_ada_count[i] / total_test_count[i]))
                logger.info('Total: Object {0} distance_1:{1}'.format(
                    i, total_distance_1_count[i] / total_test_count[i]))
                if total_unseen_objects[i] != 0:
                    if total_unseen_objects[i] - total_object_without_pose[i] != 0:
                        logger.info('Total: Unseen Object {0} add:{1} with 0.02: {2} with {3} samples '.format(
                            i, total_add_count_unseen[i] / (total_unseen_objects[i] - total_object_without_pose[i])
                            , total_add_02_count_unseen[i] / total_add_count_unseen[i],
                            (total_unseen_objects[i] - total_object_without_pose[i])))
                    logger.info(
                        'Total: Object {0} unseen :{1} times, {2} of them without poses, success rate:{3}'.format(
                            i, total_unseen_objects[i], total_object_without_pose[i],
                            (total_unseen_objects[i] - total_object_without_pose[i]) / total_unseen_objects[i]))

                total_unseen_count += total_unseen_objects[i]
                total_without_pose_count += total_object_without_pose[i]
                total_count += total_test_count[i]
                total_dis += total_test_dis[i]
                total_add += total_add_count[i]
                total_add_count_unseen_count += total_add_count_unseen[i]
                total_add_02_count_unseen_count += total_add_02_count_unseen[i]
                total_add_s += total_add_s_count[i]
                total_add_pure += total_add_pure_count[i]
                total_add_02 += total_add_02_count[i]
                total_add_s_02 += total_add_s_02_count[i]
                total_add_pure_02 += total_add_pure_02_count[i]
                total_dbd += total_dbd_count[i]
                total_drr += total_drr_count[i]
                total_ada += total_ada_count[i]
                total_distance_1 += total_distance_1_count[i]
        logger.info('total dis:{0} with {1} samples'.format(
            total_dis / total_count, total_count))
        logger.info('total add:{0} with 0.02: {1}'.format(
            total_add / total_count, total_add_02 / total_add))
        logger.info('total unseen add:{0} with 0.02: {1}'.format(
            total_add_count_unseen_count / (total_unseen_count - total_without_pose_count),
            total_add_02_count_unseen_count / total_add_count_unseen_count))
        logger.info('total add_pure:{0} with 0.02: {1}'.format(
            total_add_pure / total_count, total_add_pure_02 / total_add_pure))
        logger.info('total add_s:{0} with 0.02: {1}'.format(
            total_add_s / total_count, total_add_s_02 / total_add_s))
        logger.info('detected unseen object :{0}, failed calculate {1} poses with success rate: {2}'.format(
            total_unseen_count, total_without_pose_count,
            (total_unseen_count - total_without_pose_count) / total_unseen_count))
        logger.info('Total drr:{0}'.format(
            total_drr / total_count))
        logger.info('Total ada:{0}'.format(
            total_ada / total_count))
        logger.info('Total distance_1:{0}'.format(
            total_distance_1 / total_count))
        # end of handle global output


def calDistance(pred_r, pred_t, pred_c, return_points, gt_r, gt_t, model_x, is_sym):
    model_x = torch.tensor(model_x, dtype=torch.float64).cuda()
    how_max, which_max = torch.max(pred_c, 1)
    pred_r = pred_r[which_max[0], :, :][0]
    q = pred_r
    pred_t = pred_t[0, which_max[0], :]
    pred_r_matrix = torch.tensor(pred_r, dtype=torch.float64).cuda()
    if is_debug:
        print('pred r: ', pred_r_matrix)
        print('pred t: ', return_points[0, which_max[0], :][0] + pred_t[0])
        print('gt_r: ', gt_r.transpose(0, 1))
        print('gt_t: ', gt_t)
    pred = torch.add(torch.mm(model_x, pred_r_matrix),
                     torch.tensor(return_points[0, which_max[0], :][0] + pred_t[0], dtype=torch.float64).cuda())
    target = torch.add(torch.mm(model_x, gt_r.transpose(0, 1)), gt_t)

    if is_sym:
        target = target.transpose(1, 0).contiguous().view(3, -1)
        pred = pred.transpose(1, 0).contiguous().view(3, -1)
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1).detach() - 1)
        target = target.transpose(0, 1).contiguous()
        pred = pred.transpose(0, 1).contiguous()

    dis_vector = torch.mean(pred - target, dim=0)
    return torch.mean(torch.norm((pred - target), dim=1)), dis_vector, pred


if __name__ == '__main__':
    main()
