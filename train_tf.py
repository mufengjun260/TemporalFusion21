import argparse
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from datasets.ycb.dataset_with_non_input import PoseDataset as PoseDataset_ycb
from lib.loss import Loss
from lib.network import PoseNet
from lib.utils import setup_logger

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='ycb')
parser.add_argument('--dataset_root', type=str, default='',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--output_dir', type=str, default='', help='output dir')
parser.add_argument('--object_max', type=int, default=21, help='length of classes.txt')
parser.add_argument('--loss_stable_alpha', default=5.0, help='stable rate')

opt = parser.parse_args()


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    opt.num_objects = 21
    opt.num_points = 1000
    opt.outf = 'trained_models/ycb/' + opt.output_dir
    opt.log_dir = 'experiments/logs/ycb/' + opt.output_dir
    opt.train_dir = 'experiments/tb/ycb/' + opt.output_dir + '/train'
    opt.test_dir = 'experiments/tb/ycb/' + opt.output_dir + '/test'
    opt.repeat_epoch = 1
    if not os.path.exists(opt.outf): os.makedirs(opt.outf, exist_ok=True)
    if not os.path.exists(opt.log_dir): os.makedirs(opt.log_dir, exist_ok=True)
    if not os.path.exists(opt.train_dir): os.makedirs(opt.train_dir, exist_ok=True)
    if not os.path.exists(opt.test_dir): os.makedirs(opt.test_dir, exist_ok=True)

    opt.repeat_epoch = 1

    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects,
                        object_max=opt.object_max)
    estimator.cuda()

    isFirstInitLastDatafolder = True

    if opt.resume_posenet != '':
        psp_estimator = torch.load(
            'trained_models/ycb/pose_model_26_0.012863246640872631.pth')
        pretrained_estimator = torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet))
        estimator_dict = estimator.state_dict()

        psp_dict = {k: v for k, v in psp_estimator.items() if k.find('cnn.model') == 0}
        pretrained_dict = {k: v for k, v in pretrained_estimator.items() if k.find('cnn.model') != 0}

        estimator_dict.update(psp_dict)
        estimator_dict.update(pretrained_dict)
        estimator.load_state_dict(estimator_dict)
    else:
        psp_estimator = torch.load(
            'trained_models/ycb/pose_model_26_0.012863246640872631.pth')
        psp_dict = {k: v for k, v in psp_estimator.items() if k.find('cnn.model') == 0}
        estimator_dict = estimator.state_dict()

        estimator_dict.update(psp_dict)
        estimator.load_state_dict(estimator_dict)

    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    dataset = PoseDataset_ycb('train', opt.num_points, False, opt.dataset_root, opt.noise_trans, 'ori', False)

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                             num_workers=opt.workers)
    test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, 'ori', False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                 num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print(
        '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
            len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)

    dis_vector_last_map = {key: [] for key in range(0, opt.num_objects)}
    for i in range(0, opt.num_objects):
        dis_vector_last_map[i] = None

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        global_train_dis = 0.0

        estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                list_points, list_choose, list_img, list_target, list_model_points, list_idx, list_filename, \
                list_full_img, list_focal_length, list_principal_point, list_motion = data

                for list_index in range(len(list_points)):
                    if opt.dataset == 'ycb':
                        points, choose, img, target, model_points, idx, filename, full_img, focal_length, principal_point \
                            , motion = list_points[list_index], list_choose[list_index], list_img[list_index], \
                                       list_target[list_index], list_model_points[list_index], list_idx[list_index], \
                                       list_filename[list_index], list_full_img[list_index], list_focal_length[
                                           list_index], \
                                       list_principal_point[list_index], list_motion[list_index]
                        datafolder = filename[0].split('/')[1]
                        if isFirstInitLastDatafolder:
                            lastdatafolder = datafolder
                            isFirstInitLastDatafolder = False
                        if datafolder != lastdatafolder:
                            for i in range(0, opt.num_objects):
                                dis_vector_last_map[i] = None

                            optimizer.step()
                            optimizer.zero_grad()
                            train_dis_avg = 0
                            estimator.temporalClear(opt.object_max, opt.mem_length)
                        lastdatafolder = datafolder
                    elif opt.dataset == 'linemod':
                        list_points, list_choose, list_img, list_target, list_model_points, list_idx, list_filename = data
                        points, choose, img, target, model_points, idx, filename = list_points[0]

                    points, choose, img, target, model_points, idx = points.cuda(), \
                                                                     choose.cuda(), \
                                                                     img.cuda(), \
                                                                     target.cuda(), \
                                                                     model_points.cuda(), \
                                                                     idx.cuda()

                    pred_r, pred_t, pred_c, x_return = estimator(img, points, choose, idx, focal_length,
                                                                 principal_point, motion, True)
                    loss, dis, new_points, new_target, dis_vector = criterion(pred_r, pred_t, pred_c,
                                                                              dis_vector_last_map[idx.item()], target,
                                                                              model_points,
                                                                              idx,
                                                                              x_return,
                                                                              opt.w, False,
                                                                              float(opt.loss_stable_alpha))
                    dis_vector_last_map[idx.item()] = dis_vector
                    loss.backward(retain_graph=True)

                    logger.info('Train time {0} Frame {1} Object {2}, Loss = {3}'.format(
                        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), filename,
                        idx.item(), dis))
                    train_dis_avg += dis.item()
                    global_train_dis += dis.item()
                    train_count += 1
                    if train_count % (len(list_points) * opt.batch_size) == 0:
                        logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(
                            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch,
                            int(train_count / opt.batch_size), train_count,
                            train_dis_avg / (len(list_points) * opt.batch_size)))
                        optimizer.step()
                        optimizer.zero_grad()
                        train_dis_avg = 0

                    if train_count != 0 and train_count % 1000 == 0:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))
        global_train_dis = 0.0

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()

        for i in range(0, opt.num_objects):
            dis_vector_last_map[i] = None

        with torch.no_grad():
            isFirstInitLastDatafolder = True
            for j, data in enumerate(testdataloader, 0):
                if opt.dataset == 'ycb':
                    list_points, list_choose, list_img, list_target, list_model_points, list_idx, list_filename, \
                    list_full_img, list_focal_length, list_principal_point, list_motion = data
                for list_index in range(len(list_points)):
                    points, choose, img, target, model_points, idx, filename, full_img, focal_length, principal_point, motion \
                        = list_points[list_index], list_choose[list_index], list_img[list_index], \
                          list_target[list_index], list_model_points[list_index], list_idx[list_index], \
                          list_filename[list_index], list_full_img[list_index], list_focal_length[list_index], \
                          list_principal_point[list_index], list_motion[list_index]
                    datafolder = filename[0].split('/')[1]
                    filehead = filename[0].split('/')[2]
                    if isFirstInitLastDatafolder:
                        lastdatafolder = datafolder
                        isFirstInitLastDatafolder = False
                    if datafolder != lastdatafolder:
                        train_dis_avg = 0
                        estimator.temporalClear(opt.object_max)
                    lastdatafolder = datafolder
                    points, choose, img, target, model_points, idx = points.cuda(), \
                                                                     choose.cuda(), \
                                                                     img.cuda(), \
                                                                     target.cuda(), \
                                                                     model_points.cuda(), \
                                                                     idx.cuda()
                    cloud_path = "experiments/clouds/ycb/{0}/{1}/{2}/{3}_{4}".format(opt.output_dir, epoch,
                                                                                     datafolder, filehead,
                                                                                     int(idx))  # folder to save logs
                    if not os.path.exists("experiments/clouds/ycb/{0}/{1}/{2}".format(opt.output_dir, epoch,
                                                                                      datafolder)): os.makedirs(
                        "experiments/clouds/ycb/{0}/{1}/{2}".format(opt.output_dir, epoch,
                                                                    datafolder), exist_ok=True)
                    pred_r, pred_t, pred_c, x_return = estimator(img, points, choose, idx, focal_length,
                                                                 principal_point, motion, cloud_path)

                    _, dis, new_points, new_target, dis_vector = criterion(pred_r, pred_t, pred_c,
                                                                           dis_vector_last_map[idx.item()],
                                                                           target, model_points, idx,
                                                                           x_return,
                                                                           opt.w,
                                                                           opt.refine_start,
                                                                           float(opt.loss_stable_alpha))

                    dis_vector_last_map[idx.item()] = dis_vector

                    test_dis += dis.item()
                    logger.info('Test time {0} Test Frame No.{1} {2} {3} dis:{4}'.format(
                        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, filename,
                        idx.item(), dis))
                    test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(
            time.strftime("%d %Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            torch.save(estimator.state_dict(), '{0}/pose_model_ori_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)


if __name__ == '__main__':
    main()
