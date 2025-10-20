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
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset


from config import opt
from dataloader import get_dataloader
from CSI_diffusion.detector import csidiffusion


'''自定义损失函数'''
def masked_mse_loss(pred, target):
    # 创建 mask，非零点为 True
    mask = (target != 0).float()
    
    # 只计算非零点的误差
    loss = (mask * (pred - target) ** 2).sum() / mask.sum().clamp(min=1.0)  # 避免除0
    
    return loss

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

'''计算mIOU'''
def calculate_batch_iou(pred_boxes, true_boxes):
    """
    计算整个 batch 的平均 mIOU,只对非 0 框计算。
    输入:
        pred_boxes: [B, 6, 2]
        true_boxes: [B, 6, 2]
    输出:
        avg_iou: 平均 IOU
    """
    total_iou = 0.0
    total_valid = 0
    # pred_boxes = pred_boxes.detach().cpu().numpy()
    # true_boxes = true_boxes.detach().cpu().numpy()

    for b in range(pred_boxes.shape[0]):
        for i in range(3):  # 每个 batch 有 3 人
            gt_box = true_boxes[b, 2*i:2*i+2, :]  # [2, 2]
            pred_box = pred_boxes[b, 2*i:2*i+2, :]  # [2, 2]

            if torch.all(gt_box == 0):
                continue

            iou = calculate_iou_single(pred_box.view(-1), gt_box.view(-1))
            total_iou += iou
            total_valid += 1

    return total_iou / total_valid if total_valid > 0 else 0.0

def calculate_iou_single(pred, target):
    """
    单个框的IOU计算,输入 [4] 形式: [x1, y1, x2, y2]
    """
    # 交集
    x1 = max(pred[0], target[0])
    y1 = max(pred[1], target[1])
    x2 = min(pred[2], target[2])
    y2 = min(pred[3], target[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # 各自面积
    area_pred = max(0, (pred[2] - pred[0])) * max(0, (pred[3] - pred[1]))
    area_target = max(0, (target[2] - target[0])) * max(0, (target[3] - target[1]))

    union = area_pred + area_target - inter_area
    if union == 0:
        return 0.0
    return inter_area / union

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

    pre_keypoint = pre_keypoint.detach().cpu().numpy().reshape(-1, 3, 17, 2)
    true_keypoint = true_keypoint.detach().cpu().numpy().reshape(-1, 3, 17, 2)

    B = pre_keypoint.shape[0]
    pck_list = []

    for thresh in thresholds:
        total_correct = 0
        total_visible = 0

        for b in range(B):  # 每个样本
            for p in range(3):  # 每个人
                gt = true_keypoint[b, p]
                pred = pre_keypoint[b, p]

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


    start_time = time.time()
    for i, data in enumerate(train_dataloader):
        
        #加载数据
        csi_abs   = data['csi_abs'].float().cuda(non_blocking=True)   #torch.Size([32, 5, 3, 3, 30])
        csi_phase = data['csi_phase'].float().cuda(non_blocking=True) #torch.Size([32, 5, 3, 3, 30])
        keypoint  = data['keypoint'].float().cuda(non_blocking=True)  #torch.Size([32, 51, 2])
        box       = data['box'].float().cuda(non_blocking=True)       #torch.Size([32, 6, 2])
        #得到输入网络里的csi
        # B, T1, C1, C2, T2 = csi_abs.shape  # B=32, T1=5, C1=3, C2=3, T2=30
        # csi_abs = csi_abs.view(B, T1 * T2, C1, C2)  # [32, 150, 3, 3]
        csi_abs = csi_abs.permute(0, 2, 1, 3, 4).contiguous().view(csi_abs.shape[0], 3, 15, 30)
        csi_phase = csi_phase.permute(0, 2, 1, 3, 4).contiguous().view(csi_abs.shape[0], 3, 15, 30)
        csi = torch.cat([csi_abs, csi_phase], dim=2)  #torch.Size([32, 3, 30, 30])
        # csi = csi_abs
        # mask = torch.ones(csi.size(0), csi.size(2), csi.size(3), dtype=torch.bool).to(csi.device)
        keypoint = keypoint.view(32, 3, 17, 2)
        #print(keypoint[0])
        output = model(csi, keypoint)
        #print(output[0])
        #loss
        keypoint_loss = criterion(output, keypoint)  # 计算关键点的回归损失
        
        
        #print(keypoint_loss)
        #exit()
        #box_loss = criterion(pre_box, box)
        #loss =  0.1 * box_loss + keypoint_loss
        loss =  keypoint_loss

        #mIOU
        #avg_iou = calculate_batch_iou(pre_box.cpu(), box.cpu())
          
        #Pck@0.1-Pck@0.9
        pck_list=calculate_pck_range(output, keypoint)  #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0038510911424903724]

        optimizer.zero_grad()
        loss.backward()
        
        
        # for name, p in model.named_parameters():
        #     if p.grad is None:
        #         continue
        #     print(name, p.grad.shape, p.grad.abs().mean().item())
        # exit()
        
        optimizer.step()
         # 每 50 步打印一次
        if (i + 1) % 50 == 0 or (i + 1) == len(train_dataloader):
            print(f"Epoch [{epoch+1}/{opt.max_epoch}], Step [{i+1}/{len(train_dataloader)}], "
                  f"Loss: {loss.item():.4f}, Keypoint Loss: {keypoint_loss.item():.4f}"
                  #f" mIoU: {avg_iou:.4f}"
                  )
            
            visualizer.plot('train_loss', loss.item())
            visualizer.plot('train_keypoint_loss', keypoint_loss.item())
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
    name = time.strftime(opt.save_path + 'WIFIModel' + '_' + '%m%d_%H_%M_%S.pth')
    torch.save(model.state_dict(), name)

'''测试代码'''
def test(epoch,model, test_dataloader, criterion,visualizer,best_PCK,save_path):
    model.eval()

    total_loss = 0.0
    total_keypoint_loss = 0.0
    total_box_loss = 0.0
    total_samples = 0

    total_iou_sum = 0.0
    total_valid_count = 0

    total_pck_list = [0.0] * 9  # PCK@0.1 ~ PCK@0.9
    total_pck_batches = 0

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            # 加载数据
            csi_abs   = data['csi_abs'].float().cuda(non_blocking=True)
            csi_phase = data['csi_phase'].float().cuda(non_blocking=True)
            keypoint  = data['keypoint'].float().cuda(non_blocking=True)
            box       = data['box'].float().cuda(non_blocking=True)

            # CSI 预处理
            # B, T1, C1, C2, T2 = csi_abs.shape  # B=32, T1=5, C1=3, C2=3, T2=30
            # csi_abs = csi_abs.view(B, T1 * T2, C1, C2)  # [32, 150, 3, 3]
            csi_abs = csi_abs.permute(0, 2, 1, 3, 4).contiguous().view(csi_abs.shape[0], 3, 15, 30)
            csi_phase = csi_phase.permute(0, 2, 1, 3, 4).contiguous().view(csi_abs.shape[0], 3, 15, 30)
            csi = torch.cat([csi_abs, csi_phase], dim=2)  # [B, 3, 30, 30]
            # csi=csi_abs 
            # mask = torch.ones(csi.size(0), csi.size(2), csi.size(3), dtype=torch.bool).to(csi.device)
            
            keypoint = keypoint.view(32, 3, 17, 2)
            r_keypoint = torch.zeros_like(keypoint)
            
            r_keypoint[..., 0] = keypoint[... ,0] / 86.0
            r_keypoint[..., 1] = keypoint[... ,1] / 42.0
            output = model(csi, None)
            # output = model(csi_abs,csi_phase)
            # output[..., 0] = output[... ,0] * 86.0
            # output[..., 1] = output[... ,1] * 42.0
            
            pre_keypoint = output
            
            # Loss
            keypoint_loss = criterion(pre_keypoint, keypoint)
            #box_loss = criterion(pre_box, box)
            loss = keypoint_loss

            #mIOU
            #avg_iou = calculate_batch_iou(pre_box.cpu(), box.cpu()) #[1]
            #total_iou_sum += avg_iou * csi.shape[0]  # 乘 batch_size
            total_valid_count += csi.shape[0]
          
            #Pck@0.1-Pck@0.9
            pck_list=calculate_pck_range(pre_keypoint,keypoint)  #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0038510911424903724]
            total_pck_list = [x + y for x, y in zip(total_pck_list, pck_list)]
            total_pck_batches += 1
            
            # 累加
            batch_size = csi.shape[0]
            total_loss += loss.item() * batch_size
            total_keypoint_loss += keypoint_loss.item() * batch_size
            #total_box_loss += box_loss.item() * batch_size
            total_samples += batch_size
    
    # 平均损失
    avg_loss = total_loss / total_samples
    avg_keypoint_loss = total_keypoint_loss / total_samples
    avg_box_loss = total_box_loss / total_samples
    visualizer.plot('test_loss', avg_loss)
    visualizer.plot('test_keypoint_loss', avg_keypoint_loss)
    visualizer.plot('test_box_loss', avg_box_loss)

    # 平均 mIoU
    #avg_miou = total_iou_sum / total_valid_count if total_valid_count > 0 else 0.0
    #visualizer.plot('test_avg_mIoU', avg_miou)

    # 平均 PCK@0.1 - @0.9
    avg_pck_list = [x / total_pck_batches for x in total_pck_list]
        

    print(f"Epoch[Test][{epoch+1}/{opt.max_epoch}] "
          f"Loss: {avg_loss:.4f}, Keypoint Loss: {avg_keypoint_loss:.4f}, Box Loss: {avg_box_loss:.4f}, ")
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
    row = [epoch + 1, avg_loss, avg_keypoint_loss, avg_box_loss] + avg_pck_list
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [
                'Epoch', 'Loss', 'Keypoint Loss', 'Box Loss', 'Avg mIoU',
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
    if opt.use_gpu:
        model.cuda()

    #setp2 加载训练和测试数据集
    train_dataloader = get_dataloader(
       join(opt.train_data,'csidata'),'csi_abs', 'csi_phase',
       join(opt.train_data,'keybox'),'keypoints', 'boxes',
       batch_size= opt.batch_size ,shuffle=True,num_workers=opt.num_workers
    ) 
    test_dataloader = get_dataloader(
       join(opt.test_data,'csidata'),'csi_abs', 'csi_phase',
       join(opt.test_data,'keybox'),'keypoints', 'boxes',
       batch_size= opt.batch_size  ,shuffle=True,num_workers=opt.num_workers
    ) 

    #setp3加载损失函数和优化器
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    
    #setp4加载可视化界面
    visualizer = Visualizer(env='main')

    #训练模型
    best_PCK = 0
    for epoch in range(opt.max_epoch):
        save_path = time.strftime(opt.save_path) + f'best.pth'
        train(epoch,model,train_dataloader,criterion,optimizer,visualizer)
        best_PCK =test(epoch,model,train_dataloader,criterion,visualizer,best_PCK,save_path)
        scheduler.step()
    
if __name__ == "__main__":
    main()
