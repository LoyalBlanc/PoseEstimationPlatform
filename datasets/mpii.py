import copy
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode

from .transforms import get_affine_transform, affine_transform, flip_lr_joints

MPII = CfgNode()

MPII.BASIC = CfgNode()
MPII.BASIC.ROOT = "/mnt/DS416/datasets/MPII/data"
MPII.BASIC.TRAIN = "train"
MPII.BASIC.VALID = "valid"

MPII.IMAGE = CfgNode()
MPII.IMAGE.SIZE = [256, 256]
MPII.IMAGE.PIXEL_STD = 200

MPII.TARGET = CfgNode()
MPII.TARGET.NUM_JOINTS = 16
MPII.TARGET.SIZE = [64, 64]
MPII.TARGET.SIGMA = 2

MPII.AUGMENT = CfgNode()
MPII.AUGMENT.SCALE_FACTOR = 0.25
MPII.AUGMENT.ROTATION_FACTOR = 30
MPII.AUGMENT.FLIP_FLAG = True
MPII.AUGMENT.FLIP_PAIR = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]


class MPIIDataset(Dataset):
    def __init__(self, is_train, transform=None):
        super(MPIIDataset, self).__init__()

        self.is_train = is_train
        self.transform = transform

        # Basic
        self.root = MPII.BASIC.ROOT
        self.image_set = MPII.BASIC.TRAIN if is_train else MPII.BASIC.VALID

        # Image
        self.image_size = np.array(MPII.IMAGE.SIZE)
        self.pixel_std = MPII.IMAGE.PIXEL_STD

        # Target
        self.num_joints = MPII.TARGET.NUM_JOINTS
        self.target_size = np.array(MPII.TARGET.SIZE)
        self.sigma = MPII.TARGET.SIGMA

        # Augment
        self.scale_factor = MPII.AUGMENT.SCALE_FACTOR
        self.rotation_factor = MPII.AUGMENT.ROTATION_FACTOR
        self.flip_flag = MPII.AUGMENT.FLIP_FLAG
        self.flip_pair = MPII.AUGMENT.FLIP_PAIR

        self.db = self._get_db()
        if self.is_train:
            self.db = self.select_data(self.db)

        print('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.root, 'annot', self.image_set + '.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1, we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_visibility = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, 'joint num diff: {} vs {}'.format(len(joints), self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_visibility[:, 0] = joints_vis[:]
                joints_3d_visibility[:, 1] = joints_vis[:]

            gt_db.append(
                {
                    'image': os.path.join(self.root, 'images', image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_visibility': joints_3d_visibility,
                }
            )

        return gt_db

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_visibility']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std ** 2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2 ** 2) / (0.2 ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        print('=> num db: {}'.format(len(db)))
        print('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        c = db_rec['center']
        s = db_rec['scale']
        joints = db_rec['joints_3d']
        joints_visibility = db_rec['joints_3d_visibility']

        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if data_numpy is None:
            print('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        # Augment
        r = 0
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

            if self.flip_flag and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_visibility = flip_lr_joints(joints, joints_visibility, data_numpy.shape[1],
                                                           self.flip_pair)
                c[0] = data_numpy.shape[1] - c[0] - 1
        trans = get_affine_transform(c, s, r, self.image_size)

        # Image
        input_data = cv2.warpAffine(
            data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        # Target
        for i in range(self.num_joints):
            if joints_visibility[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        target, target_visibility = self.generate_target(joints, joints_visibility)

        # To torch
        if self.transform:
            input_data = self.transform(input_data)
        target = torch.Tensor(target)
        target_visibility = torch.Tensor(target_visibility)

        meta = {
            'image': image_file,
            'joints': joints,
            'joints_visibility': joints_visibility,
            'center': c,
            'scale': s,
            'rotation': r,
        }

        return input_data, target, target_visibility, meta

    def generate_target(self, joints, joints_vis):
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_visibility(1: visible, 0: invisible)
        """
        target_visibility = np.ones((self.num_joints, 1), dtype=np.float32)
        target_visibility[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.num_joints,
                           self.target_size[1],
                           self.target_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.target_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.target_size[0] or ul[1] >= self.target_size[1] or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_visibility[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.target_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.target_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.target_size[0])
            img_y = max(0, ul[1]), min(br[1], self.target_size[1])

            v = target_visibility[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_visibility

    def __len__(self, ):
        return len(self.db)
