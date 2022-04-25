from turtle import forward
import torch
import torch.nn as nn
import collections


class Conv(nn.Module):
    """
        a conv unit in yolov5 is   [con2d , bn , silu]  why silu ?:
    """

    def __init__(self, cin, cout, kernel_size=1, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, cin, cout, shortcut=True, e=1):
        super().__init__()
        c_ = int(cin * e)  # hidden channels
        self.cv1 = Conv(cin, c_, 1, 1, 0)
        self.cv2 = Conv(c_, cout, 1, 1, 0)
        self.shortcut = shortcut
        self.add = self.shortcut and cin == cout

    def forward(self, x):
        if self.add:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))


class C3(nn.Module):
    """
    """

    def __init__(self, cin, cout, shortcut=True, n=1, e=1):
        super().__init__()
        c_ = cin * e
        self.conv1 = Conv(cin, c_, 1, 1, 0)
        self.conv2 = Conv(cin, c_, 1, 1, 0)

        self.conv3 = Conv(2 * c_, cout, 1, 1, 0)  # conv3' input is cv1 cv2 concate

        self.m = nn.Sequential()
        for i in range(n):
            self.m.add_module("%d".format(i), Bottleneck(c_, c_, shortcut))

    def forward(self, x):
        x1 = self.m(self.conv1(x))
        x2 = self.conv2(x)
        x = self.conv3(
            torch.cat((x1, x2), dim=1)
        )  #  tensor[-1, C , H W] , concat C channel , so dim=1 ,
        return x


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class SPPF(nn.Module):
    """
         # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    """

    def __init__(self, cin, cout, kernel_size=5):
        super().__init__()
        c_ = cin // 2
        self.conv1 = Conv(cin, c_, 1, 1, 0)
        self.conv2 = Conv(c_ * 2 * 2, cout, 1, 1, 0)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat((x, y1, y2, self.m(y2)), dim=1))


class Detect(nn.Module):
    """
    anchors :  [
                [x1,y1,x2,y2,..],  #  layer0 anchors
                [x1,y1,x2,y2,..],  #  layer0 anchors
               ....]
    chs :  (inchannels_0,inchannels_1,inchannels_2,...) 每次下采样的输出层 channels tuple
    """

    def __init__(self, anchors, nc, chs=()):
        super().__init__()

        self.stride = torch.tensor([8, 16, 32], device=anchors.device)
        self.anchors = torch.zeros_like(anchors, device=anchors.device).float()
        self.anchors[0] = anchors[0] / self.stride[0]
        self.anchors[1] = anchors[1] / self.stride[1]
        self.anchors[2] = anchors[2] / self.stride[2]

        self.nc = nc  # number of classes
        self.nlayres = len(anchors)  # number of detect layers
        self.na = len(anchors[0])  #  anchors of a layer
        
        self.no = self.nc + 5  #  number output :  nums_classes +  box + confidence
        assert self.nlayres == len(chs)

        self.grid = [torch.zeros(1)] * self.nlayres  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nlayres  # init anchor grid

        self.m = nn.ModuleList()
        for c in chs:
            self.m.append(nn.Conv2d(int(c), int(self.no * self.na), 1))

    def forward(self, x):
        """
        x is list [x1,x2,x3 ...]  is sample [ 80*80 , 40*40 , 20*20 ...]         
        base size = input_h / 8;        
        example
        640/8  = 80
        640/16 = 40
        640/32 = 20        
        """
        assert self.nlayres == len(x)

        y = []

        for i in range(self.nlayres):
            x[i] = self.m[i](x[i])
            batch_size, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view((batch_size, self.na, self.no, ny, nx))
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )  # [N , anchors_num_per_layer , ny, nx , bbox_confi]

            if self.training is False:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                yop = x[i].sigmoid()  #  to [0,1]
                yop[..., 0:2] = (yop[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy ([-0.5,1.5] + gridxy) * stride
                yop[..., 2:4] = (yop[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh ([0,4])*anchors_wh

                y.append(yop.view(batch_size, -1, self.no))  # y[0].shape = []

        # y is list
        return x if self.training else (torch.cat(y, 1), x)

    def _make_grid(self, nx, ny, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid(
            [torch.arange(ny, device=d), torch.arange(nx, device=d)]
        )

        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()

        # self.anchors[i] 为一个 layer[i] 的 anchors  , 里面共有 2*n 个 wh 。 所以可以  .view(1,self.na,1,1,2)
        anchor_grid = (
            (self.anchors[i].clone() * self.stride[i])
            .view(1, self.na, 1, 1, 2)
            .expand((1, self.na, ny, nx, 2))
            .float()
        )  #  calculate real size from  layer anchors to input (640*640)

        return grid, anchor_grid


class YOLOV5S(nn.Module):
    """
        yolo v5  small  width_multiple is  0.50 of layer channel multiple
    """

    def __init__(self, imgSize, nc, anchors=None):
        super().__init__()

        assert anchors != None
        self.anchors = anchors
        self.na = len(anchors)
        self.no = nc + 5
        self.imgSize = imgSize
        self.nc = nc
        self._build_net()

    def _build_net(self):

        chs = []

        # back
        backbone_dict_8 = []
        backbone_dict_8.append(
            ("P1", Conv(3, 32, 6, 2, 2))
        )  # P1 1/2 （640-6+2*2）/2 +1  = 320  also can be focus model
        backbone_dict_8.append(
            ("P2", Conv(32, 64, 3, 2, 1))
        )  # P2 1/4 (320-3+2*1) /2 +1  = 160.5
        backbone_dict_8.append(("P2_C", C3(64, 64)))
        backbone_dict_8.append(
            ("P3", Conv(64, 128, 3, 2))
        )  # P3 1/8 (160-3+2*1) /2 +1  = 80
        backbone_dict_8.append(
            ("P3_C", C3(128, 128))
        )  # P3  channels not changed , use this to  branch_out

        backbone_dict_16 = []
        backbone_dict_16.append(
            ("P4", Conv(128, 256, 3, 2))
        )  # P4  1/16 (80 -3+ 2*1)/2 +1 = 40
        backbone_dict_16.append(("P4_C", C3(256, 256)))  # 1/16 branch_out

        backbone_dict_32 = []
        backbone_dict_32.append(
            ("P5", Conv(256, 512, 3, 2))
        )  # P5  1/16 (40 -3+ 2*1)/2 +1 = 20
        backbone_dict_32.append(
            ("P5_C3", C3(512, 512))
        )  # P5  1/16 (40 -3+ 2*1)/2 +1 = 20
        backbone_dict_32.append(
            ("P5_SPPF", SPPF(512, 512))
        )  # P5  1/16 (40 -3+ 2*1)/2 +1 = 20
        backbone_dict_32.append(
            ("P5_Tail", Conv(512, 256, kernel_size=1, stride=1, padding=0))
        )  # head_32_1   (20 -1+ 2*0)/1 +1 = 20

        # head
        head = []
        head.append(
            ("head_1_up", nn.Upsample(scale_factor=2, mode="nearest"))
        )  # 20 - >  40
        head.append(
            ("head_2_cat", Concat(dim=1))
        )  # cat 40-40 ,  channels = 256+256 : P4_C and head_1_up
        head.append(("head_3", C3(512, 256, shortcut=False)))  # c=40 channels = 256
        head.append(
            ("head_4", Conv(256, 128, kernel_size=1, stride=1, padding=0))
        )  # to catted
        head.append(
            ("head_5_up", nn.Upsample(scale_factor=2, mode="nearest"))
        )  # 40 - >  80
        head.append(
            ("head_6_cat", Concat(dim=1))
        )  # cat 80_80 , channels =  128+128=256 : head_5_up  and   P3_C  , c
        head.append(
            ("head_7", C3(256, 128, False))
        )  # size = 80 , channels = 128   # 1 to detect1 ; 2 to downsample 40

        chs.append(128)

        head.append(
            ("head_8", Conv(128, 128, kernel_size=3, stride=2, padding=1))
        )  # size =  (80-3+2*1)/2 +1 =40 , channels =128
        head.append(
            ("head_9", Concat())
        )  # cat  head_8 and  head_4  size = 40 channels = 128+128 = 256
        head.append(
            ("head_10", C3(256, 256, shortcut=False))
        )  # size = 40 , channels = 256

        chs.append(256)

        head.append(
            ("head_11", Conv(256, 256, kernel_size=3, stride=2, padding=1))
        )  # size =   (40 -3+ 2*1)/2 +1 = 20
        head.append(
            ("head_12_cat", Concat())
        )  # cat head_11 and # P5_Tail  size = 20+20=40  channesl= 2* 256 = 512
        head.append(("head_13", C3(512, 512, shortcut=False)))

        chs.append(512)
        detect = Detect(self.anchors, self.nc, chs)

        head.append(("head_14_detect", detect))

        self.backbone8 = nn.Sequential(collections.OrderedDict(backbone_dict_8))
        self.backbone16 = nn.Sequential(collections.OrderedDict(backbone_dict_16))
        self.backbone32 = nn.Sequential(collections.OrderedDict(backbone_dict_32))
        self.head = nn.Sequential(collections.OrderedDict(head))
        self.head_names_list = list(dict(self.head.named_children()).keys())

    def _get_head_model_index(self, name):
        return self.head_names_list.index(name)

    def forward(self, x):
        x0 = self.backbone8(x)  # p3
        x1 = self.backbone16(x0)  # p4
        x2 = self.backbone32(x1)  # p5

        head_x1 = self.head[self._get_head_model_index("head_1_up")](x2)
        head_x1 = self.head[self._get_head_model_index("head_2_cat")]((head_x1, x1))
        head_x1 = self.head[self._get_head_model_index("head_3")](head_x1)
        head_x1 = self.head[self._get_head_model_index("head_4")](head_x1)

        head_x2 = self.head[self._get_head_model_index("head_5_up")](head_x1)
        head_x2 = self.head[self._get_head_model_index("head_6_cat")]((head_x2, x0))

        head_x2 = self.head[self._get_head_model_index("head_7")](head_x2)
        head_x3 = self.head[self._get_head_model_index("head_8")](head_x2)
        head_x3 = self.head[self._get_head_model_index("head_9")]((head_x3, head_x1))

        head_x3 = self.head[self._get_head_model_index("head_10")](head_x3)
        head_x4 = self.head[self._get_head_model_index("head_11")](head_x3)
        head_x4 = self.head[self._get_head_model_index("head_12_cat")]((head_x4, x2))
        head_x4 = self.head[self._get_head_model_index("head_13")](head_x4)

        x = self.head[-1]([head_x2, head_x3, head_x4])

        return x
