#!/usr/bin/env python
import sys
sys.path.append("/workspace/InvisibleFace")

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

from enroll_crypto import load_enrolled_file, decrypt_sum, decode_uvw, enroll

# the ijbc dataset is from insightface
# using the cos similarity
# no flip test

# basic args
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--feat_list', type=str,
                    help='The cache folder for validation report')
parser.add_argument('--base_dir', default='/ssd/irving/data/IJB_release/IJBC/')
parser.add_argument('--type', default='c')
parser.add_argument('--embedding_size', default=512, type=int)


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


class EnrollmentFeature:
    def __init__(self, public_key='/face/irving/eval_feats/invisibleface/publickey', 
                 private_key='/face/irving/eval_feats/invisibleface/privatekey',
                 key_size=2048, K=32):
        public_key_filename = '{}_{}.npy'.format(public_key, key_size)
        self.publickey = np.load(public_key_filename, allow_pickle=True)[0]
        private_key_filename = '{}_{}.npy'.format(private_key, key_size)
        self.private_key=np.load(private_key_filename, allow_pickle=True)[0]
        self.key_size = key_size
        self.K = K
        self.L = int(np.ceil(2**(self.key_size/(2*self.K+9)-2) - 1))
        self.M = self.L/128

    def _enrollment_feature(self, feature):
        """ Enroll a feature.
        """
        result, _ = enroll(feature, self.K, self.L, self.M, self.publickey)
        c_f, C_tilde_f = result
        return c_f, C_tilde_f

    def enrollment_features(self, features):
        c_f_list, C_tilde_f_list = [], []
        for feat in features:
            c_f, C_tilde_f = self._enrollment_feature(feat)
            c_f_list.append(c_f)
            C_tilde_f_list.append(C_tilde_f)
        
        return c_f_list, C_tilde_f_list

    def enrollment_features_with_idxs(self, features, idxs):
        idx_c_f_map, idx_C_tilde_f_map = {},{}
        for i in range(len(idxs)):
            idx = idxs[i]
            feat = features[i]
            c_f, C_tilde_f = self._enrollment_feature(feat)
            idx_c_f_map[idx] = c_f
            idx_C_tilde_f_map[idx] = C_tilde_f
        
        return [idx_c_f_map, idx_C_tilde_f_map]


    def parallel_enrollment_features(self, features, num_jobs=80):
        lnum = len(features)
        idxs = list(range(0,lnum, math.ceil(lnum/num_jobs)))
        idxs.append(lnum)
        result_list = Parallel(n_jobs=num_jobs, verbose=100)(delayed(self.enrollment_features_with_idxs)(features[idxs[i]:idxs[i+1]], list(range(idxs[i], idxs[i+1]))) for i in range(num_jobs))
        
        all_idx_c_f_map, all_idx_C_tilde_f_map = {}, {}
        for (idx_c_f_map,idx_C_tilde_f_map)  in result_list:
            for idx,c_f in idx_c_f_map.items():
                all_idx_c_f_map[idx] = c_f
            for idx,C_tilde_f in idx_C_tilde_f_map.items():
                all_idx_C_tilde_f_map[idx] = C_tilde_f

        c_f_list, C_tilde_f_list = [], []
        rt_feat_list = []
        for (idx, c_f) in sorted(all_idx_c_f_map.items(),key=lambda k:k[0]):
            c_f_list.append(c_f)

        for (idx, C_tilde_f) in sorted(all_idx_C_tilde_f_map.items(),key=lambda k:k[0]):
            C_tilde_f_list.append(C_tilde_f)


        return c_f_list, C_tilde_f_list

    def distance_(self, c_x_list, C_tilde_x_list, c_y_list, C_tilde_y_list):
        num = len(c_x_list)
        dist = []
        for i in range(num):
            c_x, C_tilde_x, c_y, C_tilde_y = c_x_list[i], C_tilde_x_list[i], c_y_list[i], C_tilde_y_list[i]

            # generate bar_c_xy
            c_xy = c_x*c_y
            n = len(c_x)    
            bar_c_xy = [sum(c_xy[i:i+n//self.K]) for i in range(0, n, n//self.K)]

            # decrypt 
            C_z = decrypt_sum(C_tilde_x, C_tilde_y, self.private_key)

            # recover u_list, v_list, w
            u_list, v_list, w_z = decode_uvw(C_z, self.K, self.L)
            s_list = [1 if v%2==0 else -1 for v in v_list]

            # calculate the score
            W_z = np.e**((w_z - 2**15 * self.L**8)/(2**14 * self.L**7*self.M))
            score = W_z * sum([bar_c_xy[i]/(s_list[i] * np.e**((u_list[i]-2*self.L)/self.M)) for i in range(self.K)])

            dist.append(np.arccos(score) / math.pi)
    
        return np.array(dist)

def distance_(embeddings0, embeddings1):
    # # Distance based on cosine similarity
    # dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    # norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # # shaving
    # similarity = np.clip(dot / norm, -1., 1.)
    # dist = np.arccos(similarity) / math.pi
    # return dist
    cos = nn.CosineSimilarity(dim=1, eps=0)
    simi = torch.clamp(cos(embeddings0, embeddings1), min=-1, max=1)
    dist = torch.acos(simi)/math.pi
    return dist.cpu().numpy()


def verification(template_feats, unique_templates, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates)+1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    enm = EnrollmentFeature()

    scores = np.zeros((len(p1),))   # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    # small batchsize instead of all pairs in one batch due to the memory limiation
    batchsize = 100000
    sublists = [total_pairs[i:i + batchsize]
                for i in range(0, len(p1), batchsize)]
    all_c_f_list, all_C_tilde_f_list = enm.parallel_enrollment_features(np.array(template_feats))
    all_c_f_list = np.array(all_c_f_list)
    all_C_tilde_f_list = np.array(all_C_tilde_f_list)
    total_sublists = len(sublists)
    # import pdb;pdb.set_trace()
    for c, s in enumerate(sublists):
        c_f_list1 = all_c_f_list[template2id[p1[s]]]
        C_tilde_f_list1 = all_C_tilde_f_list[template2id[p1[s]]]
        
        c_f_list2 = all_c_f_list[template2id[p2[s]]]
        C_tilde_f_list2 = all_C_tilde_f_list[template2id[p2[s]]]
        
        dist = enm.distance_(c_f_list1, C_tilde_f_list1, c_f_list2, C_tilde_f_list2)
        scores[s] = 1-dist
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return scores


def perform_verification(args):
    # load the data
    # import ipdb;ipdb.set_trace()

    templates, medias = read_template_media_list(
        '{}/meta/ijb{}_face_tid_mid.txt'.format(args.base_dir, args.type)
    )
    p1, p2, label = read_template_pair_list(
        '{}/meta/ijb{}_template_pair_label.txt'.format(
            args.base_dir, args.type)
    )
    img_feats = read_feats(args)

    # calculate scores
    template_feats, unique_templates = image2template_feature(img_feats,
                                                              templates,
                                                              medias)
    scores = verification(template_feats, unique_templates, p1, p2)

    # show the results
    print('IJB{} 1v1 verification:\n'.format(args.type))
    fpr, tpr, _ = roc_curve(label, scores)
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    # tpr_fpr_row = []
    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    to_print = ''
    # tpr_fpr_table = PrettyTable(['Methods'] + list(map(str, x_labels)))
    # tpr_fpr_row.append(args.feat_list)
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        # print('     {} TAR.FAR{}'.format(tpr[min_index], x_labels[fpr_iter]))
        print('  {:0.4f}'.format(tpr[min_index]))
        # tpr_fpr_row.append('{:0.4f}'.format(tpr[min_index]))
        to_print = to_print + '  {:0.4f}'.format(tpr[min_index])
        # tpr_fpr_table.add_row(tpr_fpr_row)

    print(to_print)
    # print(tpr_fpr_table)


def perform_recognition(args):
    pass


def main():
    args = parser.parse_args()
    perform_verification(args)
    perform_recognition(args)


if __name__ == '__main__':
    main()