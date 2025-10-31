from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat
import time
import sys
import h5py

class CSIDataset(Dataset):
    def __init__(self, csi_abs_datas, csi_phase_datas, keypoints_labels, box_labels):
        self.csi_abs_datas = csi_abs_datas
        self.csi_phase_datas = csi_phase_datas
        self.keypoints_labels = keypoints_labels
        self.box_labels = box_labels

    def __len__(self):
        return len(self.csi_abs_datas)

    def __getitem__(self, idx):
        keypoint = self.keypoints_labels[idx].view(-1, 17, 2)
        vaild_mask = keypoint.abs().sum(dim=(1, 2)) > 0
        keypoint = keypoint[vaild_mask]
        label = torch.zeros(keypoint.shape[0], dtype=torch.long)
        
        return {
            'csi_abs': self.csi_abs_datas[idx],
            'csi_phase': self.csi_phase_datas[idx],
            'keypoint': keypoint,
            'label': label
        }

class N_CSIDataset(Dataset):
    def __init__(self, root_path):
        self.sample = []
        list_file = sorted([os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith('.txt')])
        
        with open(list_file[0], 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_path = os.path.join(root_path, 'csidata', line+'.mat')
                label_path = os.path.join(root_path, 'keybox', line+'.mat')
                self.sample.append((data_path, label_path))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        data_path, label_path = self.sample[idx]
        
        data = loadmat(data_path)
        label = loadmat(label_path)
        
        csi_abs = torch.tensor(data['csi_abs'])
        csi_phase = torch.tensor(data['csi_phase'])
        # B, T1, C1, C2, T2 = csi_abs.shape  # B=32, T1=5, C1=3, C2=3, T2=30
        csi_abs = csi_abs.permute(1, 0, 2, 3).contiguous().view(3, 15, 30)
        csi_phase = csi_phase.permute(1, 0, 2, 3).contiguous().view(3, 15, 30)
        csi = torch.cat([csi_abs, csi_phase], dim=1)

        keypoint = torch.tensor(label['keypoints']).view(-1, 17, 2)
        vaild_mask = keypoint.abs().sum(dim=(1, 2)) > 0
        keypoint = keypoint[vaild_mask]
        cls_label = torch.zeros(keypoint.shape[0], dtype=torch.long)
        
        return {
            'csi': csi,
            'keypoint': keypoint,
            'cls_label': cls_label
        }

class P_CSIDataset(Dataset):
    def __init__(self, root_path):
        self.sample = []
        list_file = sorted([os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith('.txt')])
        
        with open(list_file[0], 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_path = os.path.join(root_path, 'csi', line+'.mat')
                label_path = os.path.join(root_path, 'keypoint', line+'.npy')
                self.sample.append((data_path, label_path))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        data_path, label_path = self.sample[idx]
        
        with h5py.File(data_path, 'r') as f:
            data = f['csi_out'][:]
        label = np.load(label_path)
        data = data['real'] + 1j * data['imag']
        data = torch.from_numpy(data).float()
        data = torch.cat([data.real, data.imag], dim=0)
        data = data.permute(2, 3, 0, 1)
        csi = data.contiguous().view(3, 60, 60)
        keypoint = torch.from_numpy(label).view(-1, 14, 3)
        vaild_mask = keypoint.abs().sum(dim=(1, 2)) > 0
        keypoint = keypoint[vaild_mask]
        cls_label = torch.zeros(keypoint.shape[0], dtype=torch.long)
        
        return {
            'csi': csi,
            'keypoint': keypoint,
            'cls_label': cls_label
        }
        
def collate_fn(batch):
    batch_out = {
        'csi': [],
        'keypoint': [],
        'cls_label': []
    }

    for sample in batch:
        batch_out['csi'].append(sample['csi'])
        batch_out['keypoint'].append(sample['keypoint'])
        batch_out['cls_label'].append(sample['cls_label'])

    # 如果 csi 是固定 shape，可以直接 stack
    batch_out['csi'] = torch.stack(batch_out['csi'])

    # keypoint 与 box 不 stack，保持 list（因人数 n 不一定相同）
    return batch_out

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
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=True, collate_fn=collate_fn)

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


def new_get_dataloader(root_path, batch_size, shuffle, num_workers):
    start_time = time.time()
    

    dataset = N_CSIDataset(root_path=root_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=True, collate_fn=collate_fn)

    end_time = time.time()
    load_duration = end_time - start_time
    print('数据集的大小为：', len(dataset))
    print('csi_datas的形状为:', dataset[0]['csi'].shape)
    print('keypoints_labels的形状为:', dataset[0]['keypoint'].shape)
    print('耗时为: {:.2f} 秒'.format(load_duration))
    print("============================================")

    return dataloader