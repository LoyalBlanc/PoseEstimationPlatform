import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from scipy.io import loadmat

from .transforms import transform_preds


@torch.no_grad()
def get_max_predictions(outputs):
    assert outputs.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = outputs.shape[0]
    num_joints = outputs.shape[1]
    width = outputs.shape[3]
    heat_map_reshaped = outputs.reshape((batch_size, num_joints, -1))

    max_values, idx = torch.max(heat_map_reshaped, dim=-1)
    max_values = max_values.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    predictions = torch.tile(idx, (1, 1, 2)).float()

    predictions[:, :, 0] = (predictions[:, :, 0]) % width
    predictions[:, :, 1] = torch.floor((predictions[:, :, 1]) / width)

    predictions_mask = torch.tile(torch.greater(max_values, 0.0), (1, 1, 2)).float()

    predictions *= predictions_mask
    return predictions, max_values


def calc_dists(predictions, target, normalize):
    predictions = predictions.float()
    target = target.float()
    dists = torch.zeros((predictions.shape[1], predictions.shape[0]))
    for n in range(predictions.shape[0]):
        for c in range(predictions.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_predictions = predictions[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = torch.linalg.norm(normed_predictions - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist_cal = torch.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return torch.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


@torch.no_grad()
def accuracy(output, target):
    """
    Calculate accuracy according to PCK, but uses ground truth heat map rather than x,y locations
    First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    """
    idx = list(range(output.shape[1]))
    coordinates, max_values = get_max_predictions(output)
    target, _ = get_max_predictions(target)
    _, _, h, w = output.shape
    norm = torch.ones((coordinates.shape[0], 2)) * torch.Tensor([h, w]) / 10
    dists = calc_dists(coordinates, target, norm.to(target.device))

    acc = torch.zeros((len(idx) + 1)).to(target.device)
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    avg_acc = torch.Tensor([avg_acc]).to(target.device)
    return acc, avg_acc, cnt, coordinates, max_values


@torch.no_grad()
def get_final_predictions(outputs, center, scale):
    coordinates, max_values = get_max_predictions(outputs)
    _, _, h, w = outputs.shape
    predictions = coordinates.cpu().numpy()
    max_values = max_values.cpu().numpy()
    center = center.cpu().numpy()
    scale = scale.cpu().numpy()

    # Transform back
    for i in range(coordinates.shape[0]):
        predictions[i] = transform_preds(coordinates[i], center[i], scale[i], [w, h])

    return predictions, max_values


@torch.no_grad()
def evaluate(root, predictions, on_cuda=True):
    # convert 0-based index to 1-based index
    predictions = predictions[:, :, 0:2] + 1.0

    sc_bias = 0.6
    threshold = 0.5

    gt_file = os.path.join(root, 'annot', 'gt_valid.mat')
    gt_dict = loadmat(gt_file)
    dataset_joints = gt_dict['dataset_joints']
    jnt_missing = gt_dict['jnt_missing']
    pos_gt_src = gt_dict['pos_gt_src']
    headboxes_src = gt_dict['headboxes_src']

    pos_pred_src = np.transpose(predictions, [1, 2, 0])

    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]

    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]

    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= sc_bias
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err <= threshold), jnt_visible)
    pck_h = np.divide(1. * np.sum(less_than_threshold, axis=1), jnt_count)

    # save
    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pck_all = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err <= threshold, jnt_visible)
        pck_all[r, :] = np.divide(1. * np.sum(less_than_threshold, axis=1), jnt_count)

    pck_h = np.ma.array(pck_h, mask=False)
    pck_h.mask[6:8] = True

    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_count.mask[6:8] = True
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    name_value = [
        ('Head', pck_h[head]),
        ('Shoulder', 0.5 * (pck_h[lsho] + pck_h[rsho])),
        ('Elbow', 0.5 * (pck_h[lelb] + pck_h[relb])),
        ('Wrist', 0.5 * (pck_h[lwri] + pck_h[rwri])),
        ('Hip', 0.5 * (pck_h[lhip] + pck_h[rhip])),
        ('Knee', 0.5 * (pck_h[lkne] + pck_h[rkne])),
        ('Ankle', 0.5 * (pck_h[lank] + pck_h[rank])),
        ('Mean', np.sum(pck_h * jnt_ratio)),
        ('Mean@0.1', np.sum(pck_all[11, :] * jnt_ratio))
    ]
    name_value = OrderedDict(name_value)
    for key, value in name_value.items():
        if on_cuda:
            name_value[key] = torch.Tensor([value]).to(torch.device('cuda'))
        else:
            name_value[key] = torch.Tensor([value])

    return name_value


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name):
    batch_image = batch_image.clone()
    min_val = float(batch_image.min())
    max_val = float(batch_image.max())
    batch_image.add_(-min_val).div_(max_val - min_val + 1e-5)

    batch_size, num_joints, height, width = batch_heatmaps.shape

    grid_image = np.zeros((batch_size * height, (num_joints + 1) * width, 3), dtype=np.uint8)

    predictions, _ = get_max_predictions(batch_heatmaps)

    for i in range(batch_size):
        image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

        resized_image = cv2.resize(image, (int(width), int(height)))
        heatmaps = batch_heatmaps[i].mul(255).clamp(0, 255).byte().cpu().numpy()

        height_begin = height * i
        height_end = height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image, (int(predictions[i][j][0]), int(predictions[i][j][1])), 1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(masked_image, (int(predictions[i][j][0]), int(predictions[i][j][1])), 1, [0, 0, 255], 1)

            width_begin = width * (j + 1)
            width_end = width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

        grid_image[height_begin:height_end, 0:width, :] = resized_image

    cv2.imwrite(file_name, grid_image)
