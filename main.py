import argparse
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from datasets import get_mpii_loader
from datasets.post_process import accuracy, get_final_predictions, evaluate, save_batch_heatmaps
from models import JointsOHKMMSELoss, get_pose_net
from utils import distribute as dist


def get_args_parser():
    parser = argparse.ArgumentParser('draft', add_help=False)

    # base
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--output_dir', default='./checkpoints', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='./checkpoints/0.pth')

    # train
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch_limit', default=140, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    return parser


if __name__ == "__main__":
    # python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py
    GPU = [1, 2]
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(item) for item in GPU)

    # base
    args = get_args_parser().parse_args()
    dist.init_distributed_mode(args)
    seed = args.seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)

    # dataset
    _, train_loader = get_mpii_loader(args.batch_size, is_train=True)
    valid_set, valid_loader = get_mpii_loader(args.batch_size, is_train=False)

    # model
    model_without_ddp = get_pose_net(True).cuda()
    if os.path.isfile(args.resume):
        model_without_ddp.load_state_dict(torch.load(args.resume))
        print(f"Successfully load {args.resume}.")
    model = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True)

    criterion = JointsOHKMMSELoss(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_limit)

    for epoch in range(1, args.epoch_limit + 1):
        model.train()
        for index, source in enumerate(train_loader, 1):
            batch_images, batch_targets, batch_target_weights, batch_meta = source
            outputs = model(batch_images.cuda(non_blocking=True))
            batch_targets = batch_targets.cuda(non_blocking=True)
            batch_target_weights = batch_target_weights.cuda(non_blocking=True)

            loss = criterion(outputs, batch_targets, batch_target_weights)
            optimizer.zero_grad()
            loss.backward()
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if index % 10 == 0:
                _, avg_acc, _, _, _ = accuracy(outputs, batch_targets)
                reduced_dict_loss = dist.reduce_dict({'loss': loss})
                reduced_dict_acc = dist.reduce_dict({'acc': avg_acc})
                print(
                    f"Epoch [{epoch}/{args.epoch_limit}], Train Index [{index}/{len(train_loader)}], "
                    # f"Loss(Rank 0) {loss.item():.3f}, "
                    f"Loss(log-Avg) {torch.log(1e-16 + reduced_dict_loss['loss']).item():.3f}, "
                    # f"Grad norm {grad_total_norm.item():.3f}, "
                    # f"Acc(Rank 0) {avg_acc:.1%}, "
                    f"Acc(Avg) {reduced_dict_acc['acc'].mean():.1%}, "
                )
                save_batch_heatmaps(batch_images, outputs, os.path.join(args.output_dir, "train_prediction.jpg"))
                save_batch_heatmaps(batch_images, batch_targets, os.path.join(args.output_dir, "train_gt.jpg"))

        if dist.is_main_process():
            torch.save(model_without_ddp.state_dict(), os.path.join(args.output_dir, f"ckpt.pth"))
        lr_scheduler.step()

        model.eval()
        prediction_index = 0
        all_predictions = np.zeros((len(valid_set), valid_set.num_joints, 3), dtype=np.float32)
        with torch.no_grad():
            for index, (batch_images, batch_targets, batch_target_weights, batch_meta) in enumerate(valid_loader, 1):
                outputs = model(batch_images.cuda(non_blocking=True))
                batch_targets = batch_targets.cuda(non_blocking=True)
                batch_target_weights = batch_target_weights.cuda(non_blocking=True)
                loss = criterion(outputs, batch_targets, batch_target_weights)

                num_images = batch_images.size(0)
                predictions, max_values = get_final_predictions(outputs, batch_meta['center'], batch_meta['scale'])
                all_predictions[prediction_index:prediction_index + num_images, :, 0:2] = predictions[:, :, 0:2]
                all_predictions[prediction_index:prediction_index + num_images, :, 2:3] = max_values
                prediction_index += num_images
                if index % 10 == 0:
                    _, avg_acc, _, _, _ = accuracy(outputs, batch_targets)
                    reduced_dict_loss = dist.reduce_dict({'loss': loss})
                    reduced_dict_acc = dist.reduce_dict({'acc': avg_acc})
                    print(
                        f"Epoch [{epoch}/{args.epoch_limit}], Valid Index [{index}/{len(valid_loader)}], "
                        # f"Loss(Rank 0) {loss.item():.3f}, "
                        f"Loss(log-Avg) {torch.log(1e-16 + reduced_dict_loss['loss']).item():.3f}, "
                        # f"Grad norm {grad_total_norm.item():.3f}, "
                        # f"Acc(Rank 0) {avg_acc:.1%}, "
                        f"Acc(Avg) {reduced_dict_acc['acc'].mean():.1%}, "
                    )
                    save_batch_heatmaps(batch_images, outputs, os.path.join(args.output_dir, "valid_prediction.jpg"))
                    save_batch_heatmaps(batch_images, batch_targets, os.path.join(args.output_dir, "valid_gt.jpg"))

        eval_result = evaluate(valid_set.root, all_predictions)
        reduced_dict_eval = dist.reduce_dict(eval_result)
        print(
            f"-------- Epoch {epoch:02d} --------\n"
            f"     Head      {len(GPU)**2 * reduced_dict_eval['Head'].item():.4f}\n"
            f"     Shoulder  {len(GPU)**2 * reduced_dict_eval['Shoulder'].item():.4f}\n"
            f"     Elbow     {len(GPU)**2 * reduced_dict_eval['Elbow'].item():.4f}\n"
            f"     Wrist     {len(GPU)**2 * reduced_dict_eval['Wrist'].item():.4f}\n"
            f"     Hip       {len(GPU)**2 * reduced_dict_eval['Hip'].item():.4f}\n"
            f"     Knee      {len(GPU)**2 * reduced_dict_eval['Knee'].item():.4f}\n"
            f"     Ankle     {len(GPU)**2 * reduced_dict_eval['Ankle'].item():.4f}\n"
            f"--------------------------\n"
            f"     Mean      {len(GPU)**2 * reduced_dict_eval['Mean'].item():.4f}\n"
            f"     Mean@0.1  {len(GPU)**2 * reduced_dict_eval['Mean@0.1'].item():.4f}\n"
            f"--------------------------\n"
        )
