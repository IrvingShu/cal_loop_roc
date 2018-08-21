# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import math
import random
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
        print('fpr:' + str(fpr_draw[i]) + '\t' + 'tpr:' +str(tpr[i]) )

    print("Finished processing roc")

    return tpr, thresholds_draw


def cal_acc(same_pairs_sim_list, diff_pairs_sim_list):
    thr = np.linspace(0, 1, 100)
    positive = same_pairs_sim_list
    positive_nums = len(positive)
    negative = diff_pairs_sim_list
    negative_nums = len(negative)    

    new_positive = []
    copy_num = negative_nums // positive_nums
    for i in range(copy_num):
        new_positive.extend(positive)
    left_num = negative_nums - copy_num * positive_nums
    for i in range(left_num):
        new_positive.append(positive[i])

    new_positive_nums = len(new_positive)
    print('positive pairs: %d' % new_positive_nums)
    print('negative pairs: %d' % negative_nums)

    max_acc = 0.0
    for i in range(len(list(thr))):
        T = 0
        F = 0
        for j in range(len(new_positive)):
            if new_positive[j] >= thr[i]:
                T += 1
        for j in range(len(negative)):
            if negative[j] < thr[i]:
                T += 1
        acc = 1.0 * T / (len(new_positive) + len(negative))
        if acc > max_acc:
            max_acc = acc
            max_acc_thr = thr[i] 
    return max_acc, max_acc_thr, new_positive_nums, negative_nums   


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

    (max_acc, max_acc_thr, new_positive_nums, negative_nums) = cal_acc(same_pairs_sim_list, diff_pairs_sim_list)
    print('Acc: %f' % max_acc, 'threshold:',str(max_acc_thr))

    with open(roc_save_txt, 'w') as f:
        f.write('Save Roc: ' + '\n')
        f.write('positive pairs: %d\n' %len(same_pairs_sim_list))
        f.write('negative pairs: %d\n' %len(diff_pairs_sim_list))
        for i in range(len(fpr_draw)):
            f.write('fpr:' + str(fpr_draw[i]) + ' ' + 'tpr:' + str(tpr[i]) + ' ' + 'threshold: ' + str(thresholds_draw[i]) + '\n')
        f.write('Save Acc : ' + '\n')
        f.write('positive pairs: %d \n' % new_positive_nums)
        f.write('negative pairs: %d \n' % negative_nums)
        f.write('Acc: ' + str(max_acc)+ ' ' + 'threshold: '+ str(max_acc_thr))
       

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))

