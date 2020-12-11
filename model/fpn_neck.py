import torch.nn as nn
import torch.nn.functional as F
import math
from model.semodel import *
from model.config import DefaultConfig as cfg

class FPN(nn.Module):
    '''only for resnet50,101,152'''
    def __init__(self,features=256,use_p5=True):
        super(FPN,self).__init__()
        if cfg.backbone == 'resnet50':
            print("backbone use resnet50")
            self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
            self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        elif cfg.backbone == 'vovnet39':
            print("backbone use vovnet39")
            self.prj_5 = nn.Conv2d(1024, features, kernel_size=1)
            self.prj_4 = nn.Conv2d(768, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        # self.prj_2 = nn.Conv2d(256, features, kernel_size=1)
        self.conv_5 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(features, features, kernel_size=3, padding=1)

        # self.semodel = SE(256)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5=use_p5
        self.apply(self.init_conv_kaiming)
    def upsamplelike(self,inputs):
        src,target=inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                    mode='nearest')
    
    def init_conv_kaiming(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,x):
        C3,C4,C5=x

        # C2,C3,C4,C5=x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        # P2 = self.prj_2(C2)
        
        P4 = P4 + self.upsamplelike([P5,C4])
        P3 = P3 + self.upsamplelike([P4,C3])
        # P2 = P2 + self.upsamplelike([P3, C2])

        P3 = self.conv_3(P3)
        # P3 = self.semodel(P3)
        P4 = self.conv_4(P4)
        # P4 = self.semodel(P4)
        P5 = self.conv_5(P5)
        # P5 = self.semodel(P5)
        
        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        # P6 = self.semodel(P6)
        P7 = self.conv_out7(F.relu(P6))
        # P7 = self.semodel(P7)
        # P3 = SE(512)(P3)
        # P4 = SE(P4)
        # P5 = SE(P5)
        # P6 = SE(P6)
        # P7 = SE(P7)
        return [P3,P4,P5,P6,P7]


