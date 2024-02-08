# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import utils.util as util
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MSE_WeightLoss(nn.Module):
    def __init__(self):
        super(MSE_WeightLoss, self).__init__()

    def forward(self, input, target, hog_map):
        loss = torch.square(input - target)
        loss = torch.mean(loss, dim=2)
        loss = torch.mean(loss*hog_map, dim=(0, 1))
        return loss


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = MSE_WeightLoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos, hog = batch['data'], batch['mask'], batch['hog']
        images = images.to(device, non_blocking=True)
        hog = hog.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            images_patch = rearrange(images, 'b c (h p1) (w p2) (s p3)-> b (h w s) (p1 p2 p3 c)', p1=patch_size,
                                     p2=patch_size, p3=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)
            B, _ = hog.shape
            hog_map = hog[bool_masked_pos].reshape(B, -1)

        with torch.cuda.amp.autocast():
            outputs = model(images, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels, hog_map=hog_map)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_one_epoch(model: torch.nn.Module, data_loader: Iterable, device: torch.device, epoch: int, patch_size):
    model.eval()
    metric_logger = util.MetricLogger(delimiter="  ")
    print_freq = len(data_loader)
    loss_func = nn.MSELoss()
    header = 'Epoch: [{}]'.format(epoch)
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, bool_masked_pos = batch['data'], batch['mask']
        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            images_patch = rearrange(images, 'b c (h p1) (w p2) (s p3)-> b (h w s) (p1 p2 p3 c)', p1=patch_size,
                                     p2=patch_size, p3=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(images, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
