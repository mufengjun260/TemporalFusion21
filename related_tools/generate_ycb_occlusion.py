import argparse

import cv2
import numpy as np
from PIL import Image
import scipy.io as scio

import sys

sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/home/data1/jeremy/YCB_Video_Dataset',
                    help="dataset root dir (''YCB_Video Dataset'')")
parser.add_argument('--batch_size', default=1, help="batch size")
parser.add_argument('--n_epochs', default=1, help="epochs to train")
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help="learning rate")
parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--model_save_path', default='trained_models/', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--resume_model', default='', help="resume model name")
opt = parser.parse_args()

if __name__ == '__main__':
    global_path = 'datasets/ycb/dataset_config/test_data_temporal_6.txt'
    root = opt.dataset_root
    input_file = open(global_path)
    file_list = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        file_list.append(input_line)
    input_file.close()
    count = 0
    for path in file_list:
        label = np.array(Image.open('{0}/{1}-seg.png'.format(root, path)))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(opt.dataset_root, path))
        obj = meta['cls_indexes'].flatten().astype(np.int32)
        masked_result = np.zeros((480, 640), np.uint8)
        masked_result = label
        img = np.array(Image.open('{0}/{1}-color.png'.format(root, path)))

        end_x = count * 100
        start_x = count * 100 - 150

        if start_x > 640 or end_x < 0:
            start_x = 0
            end_x = 0
            count = 0
        else:
            count += 1

        img[0:480, start_x:end_x, :] = 0
        masked_result[0:480, start_x:end_x] = 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('{0}/{1}-color-masked-square.png'.format(root, path), img)
        cv2.imwrite('{0}/{1}-seg-masked-square.png'.format(root, path), masked_result)
        print('output file {0}/{1}-seg-masked-square'.format(root, path))
