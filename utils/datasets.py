import torch
from pathlib import Path
import  os
import  cv2
import  json
import  random
import shutil
import numpy as np
from torch.utils.data import DataLoader, Dataset
# os.chdir('../')
from utils.augmentations import  Albumentations ,letterbox,random_perspective,augment_hsv
from utils.general import xywhn2xyxy,xyxy2xywhn
import  yaml
import glob

def create_dataloader(imgsdir,labelsdir,imgsz,bs,hyp=None,augment=False,shuffle=True):        
    dataset=LoadImgandLabels(imgsdir,labelsdir,imgsz,hyp,augment)
    return DataLoader(dataset,batch_size=bs,shuffle=shuffle,num_workers=0,collate_fn=dataset.collate_fn,pin_memory=True),dataset

class  LoadImgandLabels(Dataset):
    def __init__(self,img_folder,labels_folder,imgsz,hyp,augment=False): 
        self.hyp=hyp
        assert  os.path.isdir(img_folder) and os.path.isdir(labels_folder)
        
        self.img_folder=img_folder
        self.labels_folder=labels_folder
        self.imgsz=imgsz
        self.augment=augment
        self.albumentations= Albumentations() if augment else None
        img_format=('.jpg','.bmp','.png')
        
        try:
            files = os.listdir(img_folder)
            self.img_names= [x for x in files if x.endswith(img_format)]
            
            files = os.listdir(labels_folder)            
            labels= [x for x in files if x.endswith('.txt')]
            
            if len(labels) != len(self.img_names): 
                raise Exception('image nums != labels nums')
            
            self.labels=[]
            for nam in self.img_names:
                lab= nam.split('.')[0]+'.txt'
                if lab not in labels:
                    raise Exception(  f'do not have a label for image {nam}')
                
                val = np.loadtxt(os.path.join(self.labels_folder,lab))
                self.labels.append(val)
            
        except  Exception as ex: 
            raise Exception(f'load data failed : {ex}')
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img ,(h0,w0),(h,w)=self.load_image(index)
        img, ratio, pad = letterbox(img, self.imgsz, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad) 
        labels=self.labels[index].copy()
        
        if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                
        hyp=self.hyp
        if self.augment:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            
        if self.augment:
                # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_names[index], shapes
    
    def load_image(self,i):
        path=os.path.join(self.img_folder,self.img_names[i])
        img=cv2.imread(path)
        if img is None:
            raise Exception(f'{path} not found')
        h0,w0=img.shape[:2]
        r = self.imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(
                    img,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA, )
        return img, (h0, w0), img.shape[:2]  # im, hw_original, hw_resized
   
    
    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes



IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
)  # include image suffixes
VID_FORMATS = (
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
)  # include video suffixes

class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=False):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files



def test():
    os.chdir('../')
    img_dir=f'D:\\Project\\Project_Files&Codes\\DeapLearning\\git_repos\\massage_datasets\\images\\train'
    label_dir=f'D:\\Project\\Project_Files&Codes\\DeapLearning\\git_repos\\massage_datasets\\labels\\train'
    img_size=640
    hyp=yaml.load('hyp.yaml')
    bs=4
    loader,dataset = create_dataloader(img_dir, label_dir,img_size,bs,hyp,True,True)
    
    for i in range(5):
        index=random.randint(0,len(loader))
        
        targets = loader.next()


if __name__ == '__main__':
    test()