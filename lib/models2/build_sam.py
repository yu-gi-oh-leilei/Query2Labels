from collections import OrderedDict
import os
import warnings
import math
import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

# from utils.misc import clean_state_dict, clean_body_state_dict

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class resnet101_backbone(nn.Module):
    def __init__(self, pretrain=True):
        super(resnet101_backbone,self).__init__()

        # res101 = torchvision.models.resnet101(pretrained=True)

        name = 'resnet101'
        dilation = False
        res101 = getattr(torchvision.models, name)(
                        replace_stride_with_dilation=[False, False, dilation],
                        pretrained=True,
                        norm_layer=FrozenBatchNorm2d)

        train_backbone = True
        for name, parameter in res101.named_parameters():
            if not train_backbone or 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
    
        numFit = res101.fc.in_features # 2048
        self.resnet_layer = nn.Sequential(*list(res101.children())[:-2])
        self.feat_dim = numFit

    def forward(self,x):
        feats = self.resnet_layer(x)
        return feats


class ResNet101_SAM(nn.Module):
    def __init__(self, num_classes, hidden_dim=1024):
        super(ResNet101_SAM, self).__init__()
        self.num_classes = num_classes
        self.backbone = resnet101_backbone(pretrain=True)

        self.conv_transform = nn.Conv2d(self.backbone.feat_dim, hidden_dim, (1,1))
        self.fc = nn.Conv2d(self.backbone.feat_dim, num_classes, (1,1), bias=False)
        self.group_fc = GroupWiseLinear(num_classes, hidden_dim, bias=True)

    def forward_feature(self, x):
        x = self.backbone(x)
        return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048   
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape: 
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, N, C_out) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x) # (B, N, H, W)
        mask = mask.view(mask.size(0), mask.size(1), -1) # (B, N, H*W)
        mask = torch.sigmoid(mask) # (B, N, H*W)

        x = self.conv_transform(x) # (B, C_out, H, W)
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2) # (B, H*W, C_out)
        x = torch.matmul(mask, x)
        return x
   
    def forward(self, x):
        x = self.forward_feature(x)

        out1 = self.forward_classification_sm(x)
        
        v = self.forward_sam(x) # B*num_classes*1024
        out2 = self.group_fc(v)

        out = (out1+out2)/2
        return out

class ResNet101_GMP(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101_GMP, self).__init__()

        self.backbone = resnet101_backbone(pretrain=True)
        self.num_classes = num_classes
        self.pool= nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward_feature(self, x):
        x = self.backbone(x)
        return x
   
    def forward(self, x):
        feature = self.forward_feature(x)
        feature = self.pool(feature).view(-1, 2048)
        out = self.fc(feature)
        
        return out


def build_sam(args):

    model = ResNet101_SAM(
        num_classes=args.num_class,
        hidden_dim=args.hidden_dim
    )

    return model


if __name__ == '__main__':

    model = ResNet101_SAM(num_classes=80)
    # print(model)
    # for k, v in model.state_dict().items():
    #     print(k)

    model.cuda().eval()
    image = torch.randn(4, 3, 448, 448)
    image = image.cuda()
    output1 = model(image)