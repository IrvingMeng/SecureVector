import math
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--feat_list', type=str,
                    help='feature file.')
parser.add_argument('--pair_list', type=str,
                    help='opensource pair list.')

import sys
sys.path.append("/workspace/InvisibleFace")


# NOTE: may fail to import from below python files caused by the `args`.
# from crypto_system import load_enrolled_file, decrypt_sum, decode_uvw
# from enrollment import enroll

from enroll_crypto import load_enrolled_file, decrypt_sum, decode_uvw, enroll

def load_feat_pair(feat_path, pair_path):
    pairs = {}
    with open(pair_path, 'r') as f:
        is_pair = list(map(lambda item: item.strip().split()[-1], f.readlines()))
    with open(feat_path) as f:
        ls = f.readlines()
        for idx in range(len(is_pair)):
            feat_a = ls[idx*2]
            feat_b = ls[idx*2+1]
            is_same = is_pair[idx]
            pairs[idx] = [feat_a, feat_b, is_same]
    return pairs

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

    
def calculate_roc(thresholds, embeddings0, embeddings1,
                  actual_issame, nrof_folds=10, subtract_mean=False):
    assert(embeddings0.shape[0] == embeddings1.shape[0])
    assert(embeddings0.shape[1] == embeddings1.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings0.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)
    enm = EnrollmentFeature()
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings0[train_set], embeddings1[train_set]]), axis=0)
        else:
            mean = 0.
        # NOTE: add enrollment and crypto comparison.
        # Start for enroll features & generate distance.
        c_f_list0, C_tilde_f_list0 = enm.enrollment_features(embeddings0-mean)
        c_f_list1, C_tilde_f_list1 = enm.enrollment_features(embeddings1-mean)
        
        dist = enm.distance_(c_f_list0, C_tilde_f_list0, c_f_list1, C_tilde_f_list1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

class LFold:
    def __init__(self, n_splits = 2, shuffle = False):
        self.n_splits = n_splits
        if self.n_splits>1:
            self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

    def split(self, indices):
        if self.n_splits>1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def perform_1v1_eval(feat_pairs):
    # Evaluation in the `1v1`-way.
    # feat_pairs. a tuple, shares below format.
    # [ face_a_feature, face_b_feature, gt_whether_same, face_a_calibration, face_b_calibration ]
    embeddings0 = []
    embeddings1 = []
    targets = []

    for k, v in feat_pairs.items():
        feat_a = v[0]
        feat_b = v[1]
        ab_is_same = int(v[2])

        # convert into np
        # NOTE: add normarlization for each feature.
        np_feat_a = np.asarray(feat_a.split()[1:513], dtype=float)
        np_feat_a =np_feat_a/np.linalg.norm(np_feat_a)

        np_feat_b = np.asarray(feat_b.split()[1:513], dtype=float)
        np_feat_b =np_feat_b/np.linalg.norm(np_feat_b)
        # append
        embeddings0.append(np_feat_a)
        embeddings1.append(np_feat_b)

        targets.append(ab_is_same)

    # evaluate
    embeddings0 = np.vstack(embeddings0)
    embeddings1 = np.vstack(embeddings1)
    targets = np.vstack(targets).reshape(-1,)

    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(
            thresholds, embeddings0, embeddings1,targets, subtract_mean=True)
    print('    Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    resultline='%2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy))

def eval(feat_list, pair_list):
    feat_pairs = load_feat_pair(feat_list, pair_list)
    perform_1v1_eval(feat_pairs)

def main():
    args = parser.parse_args()
    eval(args.feat_list, args.pair_list)

if __name__ == '__main__':
    main()
