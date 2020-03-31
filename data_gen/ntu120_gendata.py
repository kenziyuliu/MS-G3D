import sys
sys.path.extend(['../'])

import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm

from data_gen.preprocess import pre_normalization


# NTU RGB+D Skeleton 120 Configurations: https://arxiv.org/pdf/1905.04757.pdf
training_subjects = set([
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
])

# Even numbered setups (2,4,...,32) used for training
training_setups = set(range(2, 33, 2))

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300


def read_skeleton_filter(path):
    with open(path, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(path, max_body=4, num_joint=25):
    seq_info = read_skeleton_filter(path)
    # Create single skeleton tensor: (M, T, V, C)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # To (C,T,V,M)
    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(file_list, out_path, ignored_sample_path, benchmark, part):
    ignored_samples = []
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]

    sample_name = []
    sample_label = []
    sample_paths = []
    for folder, filename in sorted(file_list):
        if filename in ignored_samples:
            continue

        path = os.path.join(folder, filename)
        setup_loc = filename.find('S')
        subject_loc = filename.find('P')
        action_loc = filename.find('A')
        setup_id = int(filename[(setup_loc+1):(setup_loc+4)])
        subject_id = int(filename[(subject_loc+1):(subject_loc+4)])
        action_class = int(filename[(action_loc+1):(action_loc+4)])

        if benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xset':
            istraining = (setup_id in training_setups)
        else:
            raise ValueError(f'Unsupported benchmark: {benchmark}')

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError(f'Unsupported dataset part: {part}')

        if issample:
            sample_paths.append(path)
            sample_label.append(action_class - 1)   # to 0-indexed

    # Save labels
    with open(f'{out_path}/{part}_label.pkl', 'wb') as f:
        pickle.dump((sample_paths, list(sample_label)), f)

    # Create data tensor (N,C,T,V,M)
    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    # Fill in the data tensor `fp` one training example a time
    for i, s in enumerate(tqdm(sample_paths, dynamic_ncols=True)):
        data = read_xyz(s, max_body=max_body_kinect, num_joint=num_joint)
        # Fill (C,T,V,M) to data tensor (N,C,T,V,M)
        fp[i, :, 0:data.shape[1], :, :] = data

    # Perform preprocessing on data tensor
    fp = pre_normalization(fp)
    # Save input data (train/val)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D 120 Skeleton Data Extraction')
    parser.add_argument('--part1-path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--part2-path', default='../data/nturgbd_raw/nturgb+d_skeletons120/')
    parser.add_argument('--ignored-sample-path',
                        default='../data/nturgbd_raw/NTU_RGBD120_samples_with_missing_skeletons.txt')
    parser.add_argument('--out-folder', default='../data/ntu120/')

    benchmark = ['xsub', 'xset']
    part = ['train', 'val']
    arg = parser.parse_args()

    # Combine skeleton file paths
    file_list = []
    for folder in [arg.part1_path, arg.part2_path]:
        for path in os.listdir(folder):
            file_list.append((folder, path))

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(file_list, out_path, arg.ignored_sample_path, benchmark=b, part=p)

