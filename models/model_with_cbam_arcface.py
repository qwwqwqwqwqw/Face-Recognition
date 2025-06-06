# model_with_cbam_arcface.py
from models.resnet_backbone import ResNetFace      # ← 加上 models. 前缀
from utils.cbam import CBAM                        # ← cbam.py 在 utils 包里
from utils.arcface_head import ArcFace              # ← arcface_head.py 在 utils 包里

import paddle.nn as nn
import paddle.nn.functional as F
# 继承 ResNetFace，插入 CBAM
class ResNetFaceCBAM(ResNetFace):
    def __init__(self, **kw):
        super().__init__(**kw)
        # 在每个 stage 后插一个 CBAM
        for name in ['layer1','layer2','layer3','layer4']:
            stage = getattr(self,name)
            for i,blk in enumerate(stage):
                ch = blk.conv2.bn._num_features if hasattr(blk.conv2.bn,'_num_features') else blk.conv2.bn.num_features
                setattr(stage, str(i),
                        nn.Sequential(blk, CBAM(ch)))
        # 完成后，继承的 forward 不变，CBAM 已接入

class FaceRecognitionModel(nn.Layer):
    def __init__(self, num_classes, backbone_cfg):
        super().__init__()
        self.backbone = ResNetFaceCBAM(**backbone_cfg)
        self.head     = ArcFace(in_f=backbone_cfg['feature_dim'],
                                out_f=num_classes, s=64.0, m=0.5)
    def forward(self, x, label=None):
        feat = self.backbone(x)           # [N,feat_dim]
        if label is not None:
            return self.head(feat, label) # logits
        else:
            return F.normalize(feat,axis=1)
