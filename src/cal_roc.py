# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import math

import argparse


def get_same_and_diff_pairs(score_path):
    same_pairs_sim_list = []
    diff_pairs_sim_list = []

    with open(score_path) as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split('\t')
            label = line_list[-1]
            score = float(line_list[2])
            if label == '0':
                diff_pairs_sim_list.append(score)
            elif label == '1':
                same_pairs_sim_list.append(score)
            else:
                pass

    return same_pairs_sim_list, diff_pairs_sim_list


def cal_roc(same_pairs_sim_list, diff_pairs_sim_list, fpr_draw):
    diff_pairs_sim_list.sort(reverse=True)
    diff_pairs_nums = len(diff_pairs_sim_list)
    thresholds_draw = []
    for i in range(len(fpr_draw)):
        idx = fpr_draw[i] * diff_pairs_nums

        if idx >= 1:
            idx = int(math.ceil(idx))
            thresholds_idx = diff_pairs_sim_list[idx]
            thresholds_draw.append(thresholds_idx)
        else:
            thresholds_idx = diff_pairs_sim_list[0]
            thresholds_draw.append(thresholds_idx)

    num_threshs = len(thresholds_draw)

    fn = np.zeros(num_threshs)
    tp = np.zeros(num_threshs)

    print("Processing ROC")
    for sim in same_pairs_sim_list:
        for i in range(num_threshs):
            if sim < thresholds_draw[i]:
                fn[i] += 1
            else:
                tp[i] += 1
    print("Finished processing same pairs")

    tpr = tp / (tp + fn)

    for i in range(num_threshs):
        print(str(fpr_draw[i]) + '\t' + str(tpr[i]) )

    print("Finished processing roc")

    return tpr, thresholds_draw


def cal_acc(same_pairs_sim_list, diff_pairs_sim_list):
    thr = np.linspace(0, 1, 100)
    positive = same_pairs_sim_list
    positive_nums = len(positive)
    negtive = diff_pairs_sim_list[0:positive_nums]

    print('positive pairs: %d' % positive_nums)
    print('negative pairs: %d' % len(negtive))

    acc_list = []
    for i in range(len(list(thr))):
        T = 0
        F = 0
        for j in range(len(positive)):
            if positive[j] >= thr[i]:
                T += 1
        for j in range(len(negtive)):
            if negtive[j] < thr[i]:
                T += 1
        acc = 1.0 * T / (positive_nums + len(negtive))
        print(T, acc)
        acc_list.append(acc)
    
    max_acc = max(acc_list)
    return max_acc


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-list-path', type=str, help='score list path')
    parser.add_argument('--roc-save-txt', type=str, help='roc save path')
    return parser.parse_args(argv)


def main(args):
    print('===> args:\n', args)
    score_list_path = args.score_list_path
    roc_save_txt = args.roc_save_txt

    fpr_draw = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    (same_pairs_sim_list, diff_pairs_sim_list) = get_same_and_diff_pairs(score_list_path)

    tpr,thresholds_draw = cal_roc(same_pairs_sim_list, diff_pairs_sim_list, fpr_draw)

    acc = cal_acc(same_pairs_sim_list, diff_pairs_sim_list)
    print('Acc: %f' % acc)

    with open(roc_save_txt, 'w') as f:
        f.write('Save Roc: ' + '\n')
        for i in range(len(fpr_draw)):
            f.write(str(fpr_draw[i]) + ': ' + str(tpr[i]) + ' ' + 'threshold: ' + str(thresholds_draw[i]) + '\n')
        f.write('Save Acc : ' + '\n')
        f.write('Acc: ' + str(acc))


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))

