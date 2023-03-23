import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

kitti_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss

def OHEM_CE_ssc_loss(pred, target, class_weights, top_k=0.25):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none",
    )
    loss = criterion(pred, target.long())
    
    # pytorch-style mean
    # valid_mask_flatten = target.flatten() != 255
    # norm_weights = class_weights[target.flatten()[valid_mask_flatten]]
    # loss_tmp = loss.flatten()[valid_mask_flatten].sum() / norm_weights.sum()
    
    flatten_loss = loss.flatten(1)
    flatten_target = target.flatten(1)
    
    topk_losses = 0
    norm_weights = 0
    for index in range(loss.shape[0]):
        flatten_target_i = flatten_target[index]
        valid_mask = (flatten_target_i != 255)
        
        flatten_loss_i = flatten_loss[index, valid_mask]
        norm_weights_i = class_weights[flatten_target_i[valid_mask]]    
        
        topk_loss, topk_indices = torch.topk(flatten_loss_i, int(flatten_loss_i.shape[0] * top_k))
        
        topk_losses += topk_loss.sum()
        norm_weights += norm_weights_i[topk_indices].sum()
        
    loss = topk_losses / torch.clamp_min(norm_weights, 1e-4)
    
    return loss

def OHEM_CE_sc_loss(pred, target, class_weights, top_k=0.25):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    
    # binary classification
    criterion = nn.BCELoss(reduction='none')
    pred_foreground = 1 - torch.softmax(pred, dim=1)[:, 0]
    pred_foreground = pred_foreground.flatten(1)
    
    topk_losses = 0
    norm_weights = 0
    for index in range(target.shape[0]):
        flatten_target = target[index].flatten()
        valid_mask = (flatten_target != 255)
        
        pred_foreground_i = pred_foreground[index, valid_mask]
        target_foreground_i = (flatten_target[valid_mask] > 0).float()
        norm_weights_i = class_weights[target_foreground_i.long()]
        
        loss_i = criterion(pred_foreground_i, target_foreground_i) * norm_weights_i
        topk_loss, topk_indices = torch.topk(loss_i, int(loss_i.shape[0] * top_k))
        
        topk_losses += topk_loss.sum()
        norm_weights += norm_weights_i[topk_indices].sum()
        
    loss = topk_losses / torch.clamp_min(norm_weights, 1e-4)
    
    return loss


def compute_frustum_dist_loss(ssc_pred, frustums_masks, frustums_class_dists):
    bs, n_frustums = frustums_class_dists.shape[:2]
    pred_prob = F.softmax(ssc_pred, dim=1)
    batch_cnt = frustums_class_dists.sum(0)
    num_class = pred_prob.shape[1]
    
    frustum_loss = 0
    frustum_nonempty = 0
    for frus in range(n_frustums):
        frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
        prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
        prob = prob.reshape(bs, num_class, -1).permute(1, 0, 2)
        prob = prob.reshape(num_class, -1)
        cum_prob = prob.sum(dim=1)  # n_classes

        total_cnt = torch.sum(batch_cnt[frus])
        total_prob = prob.sum()
        if total_prob > 0 and total_cnt > 0:
            frustum_target_proportion = batch_cnt[frus] / total_cnt
            cum_prob = cum_prob / total_prob  # n_classes
            frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
            frustum_loss += frustum_loss_i
            frustum_nonempty += 1
    
    frustum_loss = frustum_loss / frustum_nonempty
    
    return frustum_loss