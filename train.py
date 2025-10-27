import json
import logging
import math
import time
import sys
from os.path import join                                   
import cv2
import numpy as np
import scipy.io as sio
from datetime import datetime
import os 
import visdom
import csv
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
from scipy.optimize import linear_sum_assignment


from config import opt
from dataloader import get_dataloader, new_get_dataloader
from CSI_diffusion.detector import csidiffusion

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''可视化界面'''
class Visualizer(object):
    def __init__(self, env='defualt', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='defualt', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs)
        self.index[name] = x + 1


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_keypoints=5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_keypoints = cost_keypoints

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
            pred_logits: [B, num_queries, num_classes+1]
            pred_keypoints: [B, num_queries, K, 2]
        targets:
            list of dict, each has:
                'labels': [num_gt]
                'keypoints': [num_gt, K, 2]
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].softmax(-1)  # [B, num_queries, C]
        out_kpts = outputs['pred_keypoints']

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]['labels']              # [num_gt]
            tgt_kpts = targets[b]['keypoints']          # [num_gt, K, 2]

            # 分类代价（负 log 概率）
            cost_class = -out_prob[b][:, tgt_ids]       # [num_queries, num_gt]

            # 关键点 L1 代价
            cost_kpt = torch.cdist(out_kpts[b].flatten(1), tgt_kpts.flatten(1), p=1)

            # 组合总代价
            C = self.cost_class * cost_class + self.cost_keypoints * cost_kpt
            C = C.cpu()

            pred_ind, tgt_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(pred_ind, dtype=torch.int64),
                            torch.as_tensor(tgt_ind, dtype=torch.int64)))
        return indices

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict=None, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict or {'loss_ce': 1, 'loss_kpt': 5}
        self.eos_coef = eos_coef

        # 背景类别权重
        empty_weight = torch.ones(self.num_classes + 1).to(device)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        """分类损失 (CrossEntropy)，含 no-object"""
        src_logits = outputs['pred_logits']  # [B, Q, C+1]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_keypoints(self, outputs, targets, indices):
        """关键点 L1 损失"""
        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs['pred_keypoints'][idx]           # [num_match, K, 2]
        target_kpts = torch.cat([t['keypoints'][J] for t, (_, J) in zip(targets, indices)], dim=0)
        loss_kpt = F.l1_loss(src_kpts, target_kpts, reduction='none').mean()
        return {'loss_kpt': loss_kpt}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, eva=False):
        indices = self.matcher(outputs, targets)
        losses = {}
        pck_list = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_keypoints(outputs, targets, indices))
        if eva:
            idx = self._get_src_permutation_idx(indices)
            nsrc_kpts = outputs['pred_keypoints'][idx]           # [num_match, K, 2]
            ntarget_kpts = torch.cat([t['keypoints'][J] for t, (_, J) in zip(targets, indices)], dim=0)
            pck_list=calculate_pck_range(nsrc_kpts,ntarget_kpts) 

        # 组合总损失
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys())
        losses['total_loss'] = total_loss
        return losses, pck_list

'''计算PCK'''
def calculate_pck_range(pre_keypoint, true_keypoint, thresholds=None, refer_kpts=(5, 12)):
    """
    计算整个 batch 的 PCK @ 0.1 ~ 0.9,每隔0.1。
    
    参数:
        pre_keypoint: [B, 51, 2]
        true_keypoint: [B, 51, 2]
        thresholds: list 或 array,PCK阈值列表,默认 [0.1, ..., 0.9]
        refer_kpts: 用于尺度估计的两个关键点索引（如左肩和右臀）

    返回:
        pck_list: 每个阈值下的 PCK 值列表,长度等于 thresholds 长度
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    pre_keypoint = pre_keypoint.detach().cpu().numpy().reshape(-1, 17, 2)
    true_keypoint = true_keypoint.detach().cpu().numpy().reshape(-1, 17, 2)

    B = pre_keypoint.shape[0]
    pck_list = []

    for thresh in thresholds:
        total_correct = 0
        total_visible = 0

        for b in range(B):  # 每个样本
            
            gt = true_keypoint[b]
            pred = pre_keypoint[b]

            # 计算参考尺度
            if np.all(gt[refer_kpts[0]] != 0) and np.all(gt[refer_kpts[1]] != 0):
                scale = np.linalg.norm(gt[refer_kpts[0]] - gt[refer_kpts[1]])
                if scale == 0:
                    continue
            else:
                continue

            for j in range(17):
                if np.all(gt[j] != 0):
                    dist = np.linalg.norm(pred[j] - gt[j])
                    if dist / scale <= thresh:
                        total_correct += 1
                    total_visible += 1

        pck = total_correct / total_visible if total_visible > 0 else 0.0
        pck_list.append(pck)

    return pck_list 

'''训练代码'''
def train(epoch,model,train_dataloader,criterion,optimizer,visualizer):
    model.train()
    eva_flag = False

    start_time = time.time()
    for i, data in enumerate(train_dataloader):
        #加载数据
        csi = data['csi'].float().to(device)  #torch.Size([32, 3, 30, 30])
        targets = []
        for idx in range(len(data['keypoint'])):
            targets.append({
                'labels': data['label'][idx].to(device),
                'keypoints': data['keypoint'][idx].to(device)
            })
        
        outputs = model(csi, targets)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(train_dataloader): eva_flag = True

        loss_dict, pck_list = criterion(outputs, targets, eva=eva_flag)
        eva_flag = False

        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
         # 每 50 步打印一次
        if (i + 1) % 50 == 0 or (i + 1) == len(train_dataloader):
            print(f"Epoch [{epoch+1}/{opt.max_epoch}], Step [{i+1}/{len(train_dataloader)}], "
                  f"Total Loss: {loss_dict['total_loss'].item():.4f}, Keypoint Loss: {loss_dict['loss_kpt'].item():.4f}, CLass Loss: {loss_dict['loss_ce'].item():.4f}"
                  #f" mIoU: {avg_iou:.4f}"
                  )
            visualizer.plot('train_loss', loss_dict['total_loss'].item())
            visualizer.plot('train_keypoint_loss', loss_dict['total_loss'].item())
            #visualizer.plot('train_box_loss', box_loss.item())
            #visualizer.plot('train_mIoU', avg_iou.detach().cpu().numpy())
                
            print("PCK Results:")

            for j, pck in enumerate(pck_list):
                if j ==4:
                   visualizer.plot(f'train_PCK@{(j+1)/10:.1f}', pck)
                print(f"  PCK@{(j+1)/10:.1f}: {pck:.4f}")
            
                
    #完成时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Epoch [{epoch+1}/{opt.max_epoch}] completed in {elapsed_time:.2f} seconds.")
    #保存每轮的模型
    name = opt.save_path + 'last.pth'
    torch.save(model.state_dict(), name)

'''测试代码'''
def test(epoch,model, test_dataloader, criterion,visualizer,best_PCK,save_path):
    model.eval()

    total_loss = 0.0
    total_keypoint_loss = 0.0
    total_class_loss = 0.0
    total_samples = 0

    total_valid_count = 0

    total_pck_list = [0.0] * 9  # PCK@0.1 ~ PCK@0.9
    total_pck_batches = 0

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            # 加载数据
            csi_abs   = data['csi_abs'].float().to(device)
            csi_phase = data['csi_phase'].float().to(device)
            # CSI 预处理
            # B, T1, C1, C2, T2 = csi_abs.shape  # B=32, T1=5, C1=3, C2=3, T2=30
            
            csi_abs = csi_abs.permute(0, 2, 1, 3, 4).contiguous().view(csi_abs.shape[0], 3, 15, 30)
            csi_phase = csi_phase.permute(0, 2, 1, 3, 4).contiguous().view(csi_abs.shape[0], 3, 15, 30)
            csi = torch.cat([csi_abs, csi_phase], dim=2)  # [B, 3, 30, 30]
            
            targets = []
            for idx in range(len(data['keypoint'])):
                targets.append({
                    'labels': data['label'][idx].to(device),
                    'keypoints': data['keypoint'][idx].to(device)
                })
            

            outputs = model(csi, None)
        
            loss_dict, pck_list = criterion(outputs, targets, eva=True)

            total_valid_count += csi.shape[0]
          
            #Pck@0.1-Pck@0.9
            total_pck_list = [x + y for x, y in zip(total_pck_list, pck_list)]
            total_pck_batches += 1
            
            # 累加
            batch_size = csi.shape[0]
            total_loss += loss_dict['total_loss'].item() * batch_size
            total_keypoint_loss += loss_dict['loss_kpt'].item() * batch_size
            total_class_loss += loss_dict['loss_ce'].item() * batch_size
            total_samples += batch_size
    
    # 平均损失
    avg_loss = total_loss / total_samples
    avg_keypoint_loss = total_keypoint_loss / total_samples
    avg_class_loss = total_class_loss / total_samples
    visualizer.plot('test_loss', avg_loss)
    visualizer.plot('test_keypoint_loss', avg_keypoint_loss)
    visualizer.plot('test_class_loss', avg_class_loss)

    # 平均 PCK@0.1 - @0.9
    avg_pck_list = [x / total_pck_batches for x in total_pck_list]
        

    print(f"Epoch[Test][{epoch+1}/{opt.max_epoch}] "
          f"Loss: {avg_loss:.4f}, Keypoint Loss: {avg_keypoint_loss:.4f}, Class Loss: {avg_class_loss:.4f}, ")
    print("PCK Results:")
    for j, pck in enumerate(avg_pck_list):
        if j ==4:
           visualizer.plot(f'test_PCK@{(j+1)/10:.1f}', pck)
        print(f"  PCK@{(j+1)/10:.1f}: {pck:.4f}")

    
    csv_file = opt.result_file
    # 检查路径是否是目录（如果是，自动添加文件名）
    if os.path.isdir(csv_file):
       csv_file = os.path.join(csv_file, "results.csv")  
       print(f"Warning: result_file is a directory, saving to {csv_file} instead.")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.isfile(csv_file)
    row = [epoch + 1, avg_loss, avg_keypoint_loss, avg_class_loss] + avg_pck_list
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [
                'Epoch', 'Loss', 'Keypoint Loss', 'Class Loss', 
                'PCK@0.1', 'PCK@0.2', 'PCK@0.3', 'PCK@0.4', 'PCK@0.5',
                'PCK@0.6', 'PCK@0.7', 'PCK@0.8', 'PCK@0.9'
            ]
            writer.writerow(header)
        writer.writerow(row)
    

    # 判断是否是最好的 PCK@0.5 或 mIoU
    save_model = False
    if avg_pck_list[4] > best_PCK:  # PCK@0.5
        best_PCK = avg_pck_list[4]
        save_model = True
    if save_model:
        # 保存模型
        torch.save(model.state_dict(), save_path)

    return best_PCK

   
def main(**kwargs):
    print('start,开始执行代码')
    opt.parse(kwargs)
    #setp1 加载模型
    model =csidiffusion()
    print(model)
    # model = nn.DataParallel(model) 
    if opt.load_model_path:  
        model.load_state_dict(torch.load(opt.load_model_path))
    model.to(device)

    #setp2 加载训练和测试数据集
    # train_dataloader = get_dataloader(
    #    join(opt.train_data,'csidata'),'csi_abs', 'csi_phase',
    #    join(opt.train_data,'keybox'),'keypoints', 'boxes',
    #    batch_size= opt.batch_size ,shuffle=True,num_workers=opt.num_workers
    # ) 
    # test_dataloader = get_dataloader(
    #    join(opt.test_data,'csidata'),'csi_abs', 'csi_phase',
    #    join(opt.test_data,'keybox'),'keypoints', 'boxes',
    #    batch_size= opt.batch_size  ,shuffle=True,num_workers=opt.num_workers
    # ) 
    
    train_dataloader = new_get_dataloader(
       opt.train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
    ) 
    test_dataloader = new_get_dataloader(
       opt.test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
    ) 
    #setp3加载损失函数和优化器
    matcher = HungarianMatcher(cost_class=1, cost_keypoints=5)
    criterion = SetCriterion(num_classes=1, matcher=matcher)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    
    #setp4加载可视化界面
    visualizer = Visualizer(env='main')

    #训练模型
    best_PCK = 0
    for epoch in range(opt.max_epoch):
        save_path = time.strftime(opt.save_path) + f'best.pth'
        train(epoch,model,train_dataloader,criterion,optimizer,visualizer)
        best_PCK =test(epoch,model,test_dataloader,criterion,visualizer,best_PCK,save_path)
        scheduler.step()
    
if __name__ == "__main__":
    main()
