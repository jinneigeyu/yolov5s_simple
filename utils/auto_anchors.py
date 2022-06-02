from re import L
import numpy as np 
import os
import cv2
from utils.general import xywh2xyxy
import torch

def generate_anchors_kmeans(dataset,n=9,imgsize=640,device='cuda'):
    
    from scipy.cluster.vq import kmeans
    
    shapes=[]
    
    for i , img in enumerate(dataset.img_names):
        path=os.path.join(dataset.img_folder,img)
        shape = cv2.imread(path).shape[0:2][::-1]
        shapes.append(shape)
    
    # shapes=np.array(shapes)
    # labels=np.array(dataset.labels)[:,:,1:]
    # shapes=imgsize*shapes/np.max(shapes, axis = 1)[:,None]
    # wh0=np.concatenate([l[:, 2:] * s for s, l in zip(shapes,labels)])
    
    assert  len(shapes) == len(dataset.labels);
    
    wh0=[];
    for i in range(len(shapes)):
        s=shapes[i]
        l=dataset.labels[i]
        
        if l.size == 0: 
            continue
        
        start =l.shape[1]-2
        l=l[:,start:]
        wh = l*s
        wh0.append(wh)
    wh0=  np.concatenate(wh0);
           
    

    
    
    wh = wh0[(wh0 >= 2.0).any(1)]
    s = wh.std(0)  # sigmas for whitening
    k = kmeans(wh / s, n, iter=30)[0] * s  # points
    k=k[(k[:,0]*k[:,1]).argsort()] # 按照面积升序
    return torch.tensor(k,device=device)
   
        