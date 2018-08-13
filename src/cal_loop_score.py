import os
import sys
import matio
import argparse
import numpy as np

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-root-folder', type=str, help='feature root folder')
    parser.add_argument('--gallery-list-path', type=str, help='feature list path')
    parser.add_argument('--probe-list-path', type=str, help='feature list path')
    parser.add_argument('--score-save-path', type=str, help='score saved path')
    
    return parser.parse_args(argv)

def read_feature(fea_root, path):
    fea_dict = dict()
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            full_path = os.path.join(fea_root, line.strip())
            x_vec = matio.load_mat(full_path).flatten()
            fea_dict[line.strip()] = x_vec
    return fea_dict

def cal_loop_sim(gallery_fea, probe_fea):
    result = []
    for key in probe_fea:
        for key2 in gallery_fea:
            sim = np.dot(probe_fea[key], gallery_fea[key2]) / (np.linalg.norm(probe_fea[key], ord=2) * np.linalg.norm(gallery_fea[key2], ord=2))
            probe_label = key.split('/')[1]
            gallery_label = key2.split('/')[1]
            if probe_label == gallery_label:
                result.append(key + '\t' + key2 + '\t' + str(sim) + '\t' + '1' + '\n')
            else:
                result.append(key + '\t' + key2 + '\t' + str(sim) + '\t' + '0' +'\n')
    return result

def main(args):
    print('===> args:\n', args)
    fea_root_folder = args.feature_root_folder
    gallery_list_path = args.gallery_list_path
    probe_list_path = args.probe_list_path
    score_save_path = args.score_save_path

    gallery_fea_dict = read_feature(fea_root_folder, gallery_list_path)
    probe_fea_dict = read_feature(fea_root_folder, probe_list_path)
    result = cal_loop_sim(gallery_fea_dict, probe_fea_dict)
    with open(score_save_path, 'w') as f:
        for i in range(len(result)):
            f.write(result[i])

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
