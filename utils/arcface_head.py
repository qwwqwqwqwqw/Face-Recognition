# arcface_head.py
import paddle
import paddle.nn as nn, paddle.nn.functional as F, math

class ArcFace(nn.Layer):
    def __init__(self, in_f, out_f, s=64.0, m=0.5, easy=False):
        super().__init__()
        self.w = self.create_parameter([out_f,in_f],
                  default_initializer=nn.initializer.XavierUniform())
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi-m)
        self.mm = math.sin(math.pi-m)*m
        self.easy = easy

    def forward(self, x, label):
        x = F.normalize(x,axis=1)
        w = F.normalize(self.w,axis=1)
        cos = paddle.matmul(x, w, transpose_y=True)
        sin = paddle.sqrt(1 - cos**2)
        phi = cos*self.cos_m - sin*self.sin_m
        if self.easy:
            phi = paddle.where(cos>0, phi, cos)
        else:
            phi = paddle.where(cos>self.th, phi, cos-self.mm)
        one = F.one_hot(label, num_classes=self.w.shape[0]).astype('float32')
        logits = one*phi + (1-one)*cos
        return logits*self.s
