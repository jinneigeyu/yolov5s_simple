import  os
import argparse
from statistics import mode
import warnings
from tqdm import tqdm
import torch
from  models.yolo import YOLOV5S
from utils.Loss import Loss
from  utils.datasets import  create_dataloader
from utils.general import  generate_anchors
from utils.auto_anchors import generate_anchors_kmeans
import val
from tensorboardX import SummaryWriter
import yaml
from torch.cuda import amp
from utils.general import LOGGER
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

IMG_DIR='../git_repos/massage_datasets/images/train/'
ANNOS_DIR='../git_repos/massage_datasets/labels/train/'
IMG_DIR_VAL='../git_repos/massage_datasets/images/val/'
ANNOS_DIR_VA='../git_repos/massage_datasets/labels/val/'
IMG_SZ=640

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir_train', type=str, default=IMG_DIR, help='train images folder')    
    parser.add_argument('--annos_dir_train', type=str, default=ANNOS_DIR, help='train annotations folder')
    parser.add_argument('--img_dir_valid', type=str, default=IMG_DIR_VAL, help='valid images folder')    
    parser.add_argument('--annos_dir_valid', type=str, default=ANNOS_DIR_VA, help='valid annotations folder')    
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--hyp', type=str, default='hyp.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixe)')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--auto_anchor', default=True, help='use kmeans to generate anchors')
    opt =parser.parse_args()
    for i in vars(opt).items():
        print(f'{i[0]} : {i[1]}')
    return opt


def write_infos(losses,pr_aps,lr,file='result.csv',createNew=False):
    
    def write_content(f,contents):
        f.seek(0,2)
        f.writelines( contents )
         
    content=" %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.6f \n" % (losses[0].item(),losses[1].item(),losses[2].item(),pr_aps[0],pr_aps[1],pr_aps[2],pr_aps[3],lr)
     
    if os.path.exists(file) and createNew:
        os.remove(file)
        with open(file,"a") as f:
            header='box_loss \t obj_loss \t cls_loss \t Precision \t Recall \t mAP@0.5 \t mAP@0.5_.95 \t lr \n'
            write_content(f, [header,content])
        return
        
    if os.path.exists(file):        
        with open(file,'a') as f:            
            f.writelines( [content] )
    else : 
        with open(file,"a") as f:
            header='Precision \t Recall \t mAP_0.5 \t mAP_0.50.95'
            write_content(f, [header,content])
        


def train(opt):
    with open(opt.hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    
    device=opt.device                
    epochs=opt.epochs
   
    train_loader,dataset = create_dataloader(opt.img_dir_train,opt.annos_dir_train,IMG_SZ,opt.batch_size,hyp,augment=True,shuffle=True)  
    val_loader,val_dataset=create_dataloader(opt.img_dir_valid,opt.annos_dir_valid,IMG_SZ,opt.batch_size,hyp,augment=False,shuffle=False)
    
    
    anchors= generate_anchors(hyp['anchors'],device)
    
    if opt.auto_anchor:
        auto_anchors = generate_anchors_kmeans(dataset,device=opt.device)
        hyp['auto_anchors']=auto_anchors.cpu().numpy().reshape(3,-1).tolist()
        with open(opt.hyp, 'w') as f:
            f.write(yaml.dump(hyp))
        anchors= generate_anchors(hyp['auto_anchors'],device)
    # model
    model=YOLOV5S(imgSize=IMG_SZ,nc=hyp['num_class'],anchors=anchors)
    
    # optimizer
    optimizer= torch.optim.SGD(model.parameters(),lr=hyp['lr'],momentum=hyp['momentum'])
    lf= lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lr"]) + hyp["lr"]
    
    # lr scheduler
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lf)
    
    # loss
    compute_loss = Loss(anchors=(model.head[-1].anchors),hyp=hyp)
    
    # pre_trained
    pre_trained= None if opt.weights is None else True   
    if pre_trained:
        model.load_state_dict(torch.load(hyp.weights))
    
    
    cuda = device != 'cpu'
    # amp acc
    scaler = amp.GradScaler(enabled=cuda)
    if cuda :
        model.cuda()
    
    # tensorboard gragp
    fake_img = torch.randn(1, 3, 640, 640)
    fake_img = fake_img.to(device)
    writer = SummaryWriter('./log')
   
    writer.add_graph(model, fake_img)

    C=True
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        mloss = torch.zeros(3, device=device)
        LOGGER.info(("\n" + "%10s" * 7)% ("Epoch", "lr", "box", "obj", "cls", "labels", "img_size"))
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        
       
        
        # train
        for i,(imgs,targets,path,_) in pbar: 
            imgs = imgs.to(device).float()/255
            
            with amp.autocast(enabled=True):
                pred = model(imgs)
                loss , loss_items= compute_loss(pred, targets.to(device))
            
            # 1、Scales loss.  先将梯度放大 防止梯度消失
            scaler.scale(loss).backward()
            # 2、scaler.step()   再把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）           
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # loss details
            mloss= (mloss*i + loss_items)/(i+1)
            pbar.set_description(("%10s" * 2 + "%10.6g" * 5)% (f"{epoch}/{epochs - 1}",
                                                               round(scheduler.optimizer.param_groups[0]["lr"], 6),
                                                               *mloss,
                                                               targets.shape[0],
                                                               imgs.shape[-1],))
        
        scheduler.step()
        # valid
        val_results=val.run(val_loader,model,compute_loss,'cuda')        
        
        # LOGGER.info(("\n" + "%10s" * 7)% ("", "", "", {*val_results[0:4]}))
        print('\t\t\t\t\t\t %.3f      %.3f     %.3f     %.3f' % (val_results[0],val_results[1],val_results[2],val_results[3]))   
        model.float()
        torch.cuda.empty_cache()
        write_infos(mloss,val_results,scheduler.optimizer.param_groups[0]["lr"],createNew= C)
        C=False
        if epoch == 80 :
             torch.save(model.state_dict(), "save_80.pt")
            
    torch.save(model.state_dict(), "save_100.pt")
def main():
    opt=parse_opt()
    train(opt)
    
if __name__ == "__main__":
    main()