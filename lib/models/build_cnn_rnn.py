from collections import OrderedDict
import os
import warnings

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

import models
import torchvision.models as models

from utils.misc import clean_state_dict, clean_body_state_dict

'''
https://github.com/AmrMaghraby/CNN-RNN-A-Unified-Framework-for-Multi-label-Image-Classification
'''

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


class ResNet101_GAP(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101_GAP, self).__init__()

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


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        # resnet = models.resnet152(pretrained=True)
        self.backbone = resnet101_backbone(pretrain=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids



def build_baseline(args):

    # model = ResNet101_GMP(
    #     num_classes=args.num_class
    # )

    model = ResNet101_GAP(
        num_classes=args.num_class
    )

    return model