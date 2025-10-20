from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import scipy.io as sio
import time
import sys

class CSIDataset(Dataset):
    def __init__(self, csi_abs_datas, csi_phase_datas, keypoints_labels, box_labels):
        self.csi_abs_datas = csi_abs_datas
        self.csi_phase_datas = csi_phase_datas
        self.keypoints_labels = keypoints_labels
        self.box_labels = box_labels

    def __len__(self):
        return len(self.csi_abs_datas)

    def __getitem__(self, idx):
        return {
            'csi_abs': self.csi_abs_datas[idx],
            'csi_phase': self.csi_phase_datas[idx],
            'keypoint': self.keypoints_labels[idx],
            'box': self.box_labels[idx]
        }

def get_dataloader(csidata_path, data_name1, data_name2, heatmap_path, label_name1, label_name2, batch_size, shuffle, num_workers):
    start_time = time.time()
    dataset_name = os.path.basename(os.path.dirname(csidata_path))

    csi_files = sorted([os.path.join(csidata_path, f) for f in os.listdir(csidata_path) if f.endswith('.mat')])
    heatmap_files = sorted([os.path.join(heatmap_path, f) for f in os.listdir(heatmap_path) if f.endswith('.mat')])

    csi_abs_datas = []
    csi_phase_datas = []
    keypoints_labels = []
    box_labels = []

    for csi_file, heatmap_file in zip(csi_files, heatmap_files):
        csi_name = os.path.splitext(os.path.basename(csi_file))[0]
        heatmap_name = os.path.splitext(os.path.basename(heatmap_file))[0]
        if csi_name != heatmap_name:
            print(f"Error: File names do not match: {csi_name} and {heatmap_name}")
            sys.exit()

        data_dict = sio.loadmat(csi_file)
        label_dict = sio.loadmat(heatmap_file)

        data1 = data_dict[data_name1]
        data2 = data_dict[data_name2]
        label1 = label_dict[label_name1]
        label2 = label_dict[label_name2]

        csi_abs_datas.append(data1)
        csi_phase_datas.append(data2)
        keypoints_labels.append(label1)
        box_labels.append(label2)

    csi_abs_datas = torch.from_numpy(np.array(csi_abs_datas)).float()
    csi_phase_datas = torch.from_numpy(np.array(csi_phase_datas)).float()
    keypoints_labels = torch.from_numpy(np.array(keypoints_labels)).float()
    box_labels = torch.from_numpy(np.array(box_labels)).float()

    dataset = CSIDataset(csi_abs_datas, csi_phase_datas, keypoints_labels, box_labels)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=True)

    end_time = time.time()
    load_duration = end_time - start_time
    print(f"{dataset_name.capitalize()} dataset loading time: {load_duration:.2f} seconds")
    print('数据集的内容为：')
    print('csi_abs_datas的形状为:', csi_abs_datas.shape)
    print('csi_phase_datas的形状为:', csi_phase_datas.shape)
    print('keypoints_labels的形状为:', keypoints_labels.shape)
    print('box_labels的形状为:', box_labels.shape)
    print("============================================")

    return dataloader
