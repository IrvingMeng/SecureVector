#!/usr/bin/env python

import argparse
import os
import cv2
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from sklearn.metrics import roc_curve, auc
# from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from joblib import Parallel, delayed



# basic args
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--feat_list', type=str,
                    help='The cache folder for validation report')
parser.add_argument('--base_dir', default='/ssd/irving/data/IJB_release/IJBC/')
parser.add_argument('--type', default='c')
parser.add_argument('--embedding_size', default=512, type=int)
parser.add_argument('--template_feature', type=str)
parser.add_argument('--pair_list', type=str)

def read_template_media_list(path):
    ijb_meta, templates, medias = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        ijb_meta.append(parts[0])
        templates.append(int(parts[1]))
        medias.append(int(parts[2]))
    return np.array(templates), np.array(medias)


def read_template_pair_list(path):
    t1, t2, label = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(' ')
        t1.append(int(data[0]))
        t2.append(int(data[1]))
        label.append(int(data[2]))
    return np.array(t1), np.array(t2), np.array(label)


def read_feats(args):
    with open(args.feat_list, 'r') as f:
        lines = f.readlines()
    img_feats = []
    for line in lines:
        data = line.strip().split(' ')
        img_feats.append([float(ele) for ele in data[1:1+args.embedding_size]])
    img_feats = np.array(img_feats).astype(np.float32)
    return img_feats


def image2template_feature(img_feats=None,
                           templates=None,
                           medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    # template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    template_feats = torch.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]

        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(
            face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            media_norm_feats += [np.mean(face_norm_feats[ind_m],
                                         0, keepdims=False)]

        # media_norm_feats = np.array(media_norm_feats)
        media_norm_feats = torch.tensor(media_norm_feats)
        media_norm_feats = F.normalize(media_norm_feats)
        # template_feats[count_template] = np.mean(media_norm_feats, 0)
        template_feats[count_template] = torch.mean(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_feats, unique_templates


def gather_pair_features(args):

    templates, medias = read_template_media_list(
        '{}/meta/ijb{}_face_tid_mid.txt'.format(args.base_dir, args.type)
    )
    p1, p2, label = read_template_pair_list(
        '{}/meta/ijb{}_template_pair_label.txt'.format(
            args.base_dir, args.type)
    )
    img_feats = read_feats(args)
    template_feats, unique_templates = image2template_feature(img_feats,
                                                              templates,
                                                              medias)

    template2id = np.zeros((max(unique_templates)+1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    with open(args.template_feature, 'w') as ff:
        for i,feat in enumerate(template_feats):
            featlist = [str(b) for b in feat.tolist()]
            ff.write('{} {}\n'.format(i,' '.join(featlist)))

    with open(args.pair_list, 'w') as pf:
        for i in range(len(p1)):
            issame = label[i]
            pf.write('{} {} {}\n'.format(template2id[p1[i]], template2id[p2[i]], issame))


def main():
    args = parser.parse_args()
    gather_pair_features(args)


if __name__ == '__main__':
    main()