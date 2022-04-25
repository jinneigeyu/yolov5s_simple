import torch
from utils.datasets import create_dataloader
from  utils.general import  LOGGER,non_max_suppression,box_iou,xywh2xyxy,scale_coords
from utils.metrics import box_iou ,ap_per_class,compute_ap
from tqdm import tqdm
import numpy as np
from pathlib import Path

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device) # 0.5~0.95 10个iou
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    else:
        correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device) # 0.5~0.95 10个iou
    return correct


def calculate_ap(correct,confidence,pcls,tcls,eps=1e-5): #
    sort_i = np.argsort(-confidence) # 置信度降序排序
    tp, conf, pred_cls = correct[sort_i], confidence[sort_i], pcls[sort_i]    
    unique_classes,nt=np.unique(tcls,return_counts=True)  
    nc = unique_classes.shape[0] # 类别共有几种
    ap, p, r = np.zeros((nc, tp.shape[1])) , np.zeros((nc, 1000)) , np.zeros((nc, 1000))   # 每种类别建立矩阵
    px, py = np.linspace(0, 1, 1000), []  # for plotting
     
    for ci,cls_i in enumerate(unique_classes):         
        nl=nt[ci] #此类别类别的真值个数
        i=pred_cls==cls_i #选出此类别的索引
        tp_i=np.count_nonzero(i) # 此类别tp个数
        if tp_i:
            fpc = (1 - tp[i]).cumsum(0)  # 沿着 0轴方向累加到最后 ，tp 存储的是bool值， 判定大于 0.5~0.95 个iou
            tpc = tp[i].cumsum(0)            
      
            recall = tpc /(nl+eps)            
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
            
            precision = tpc/(fpc+tpc)
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
            
            # ap pr 曲线下的面积
            for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
               
    f1 = 2 * p * r / (p + r + eps)   
    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')

@torch.no_grad()
def run(val_loader,model,compute_loss,device,iou_thres=0.6,conf_thres=0.001,half=False):
    
    
    model.eval()
    cuda = device !='cpu'
    iouv=torch.linspace(0.5,0.95,10,device=device)  # 0.5 ~ 0.95 mAP
    niou = iouv.numel()
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pbar= tqdm(val_loader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') 
    loss = torch.zeros(3, device=device)
    stats=[]
    ap=[]
    ap_class=[]
    
    for batch_i,(im,targets,paths,shapes) in enumerate(pbar):
        targets= targets.to(device)        
        im = im.to(device).float()
        im /= 255
        nb,_,height,width = im.shape        
        out,train_out=model(im)
        if compute_loss:
            loss+=compute_loss([x.float() for x in train_out], targets)[1]
        
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        # out: 极大值抑制 筛选出 iou>0.5的 [xyxy,conf,cls] 
        out = non_max_suppression(out,conf_thres,iou_thres,multi_label=True,max_det=300)
        
        
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:] # 从 一个batch的targets中选出当前的
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class # 找出一个图片中 gt 存在的类别 ，tcls 是可能含有重复的
            path, shape = Path(paths[si]), shapes[si][0]

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool,device=device)
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels # 组装 此一个图片的target 与 pred 排列一致
                correct = process_batch(predn, labelsn, iouv)
                
                # iou = box_iou(labelsn[:, 1:], predn[:, :4])
                # x = torch.where((iou >= iouv[0]) & (labelsn[:, 0:1] == predn[:, 5]))  #  找到 预测值中 大于阈值 且 类别也判定正确的
                #  # 如何保证 一个 gt 只被考虑一次？？
                # if len(x[0]):
                #     label_and_index= torch.stack(x,1)
                #     iou= iou[x]
                #     matches = torch.cat((label_and_index,iou[:,None]),1).cpu().numpy()
                #     matches = matches[matches[:,2].argsort()][::-1] # 根据 iou 降序排列，选出mathches
                    
                #     i= np.unique(matches[:,1],return_index=True)[1] # 选出不重复的IOU索引 ?? 为什么要筛选呢？？          
                #     matches=matches[i]
                    
                #     i= np.unique(matches[:,0],return_index=True)[1] # 选出不重复的类别索引 ?? 为什么要筛选呢？？  , 因为要保证一个gt 只配预测一次，取那个最大的iou的值        
                #     matches=matches[i]              # 只选取每一个label的最大值,因为目前label的值是通过gt 索引的个数决定的
                    
                #     matches=torch.from_numpy(matches).to(iouv.device)
                #     correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
                    
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
        
       
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class=calculate_ap(*stats) 
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    
   
    return (mp, mr, map50, map, *(loss.cpu() / len(val_loader)).tolist())

@torch.no_grad()
def run22(dataloader,
        model,  # model.pt path(s)
        batch_size=4,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        device='cuda',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=0,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        ):

    # Configure
    model.eval()
    device= 'cuda'
   
   
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        
        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        
        # Inference
        out, train_out = model(im) 

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True, agnostic=single_cls)
        

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:] # 从 一个batch的targets中选出当前的
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class # 找出一个图片中 gt 存在的类别
            path, shape = Path(paths[si]), shapes[si][0]
            # seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels # 组装 此一个图片的target 与 pred 排列一致
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir='.', names=())
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    # Return results
    model.float()  # for training
   
