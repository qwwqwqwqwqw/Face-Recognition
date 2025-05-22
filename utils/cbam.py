# cbam.py
import paddle
import paddle.nn as nn, paddle.nn.functional as F

class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2D(1)
        self.max = nn.AdaptiveMaxPool2D(1)
        self.fc  = nn.Sequential(
            nn.Conv2D(in_planes, in_planes//r, 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(in_planes//r, in_planes,   1, bias_attr=False),
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = self.fc(self.avg(x))
        m = self.fc(self.max(x))
        return self.sig(a+m)

class SpatialAttention(nn.Layer):
    def __init__(self, k=7):
        super().__init__()
        pad = (k-1)//2
        self.conv = nn.Conv2D(2,1,k,padding=pad,bias_attr=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = paddle.mean(x,axis=1,keepdim=True)
        max = paddle.max(x,axis=1,keepdim=True)
        return self.sig(self.conv(paddle.concat([avg,max],axis=1)))

class CBAM(nn.Layer):
    def __init__(self, ch, r=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention(k)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
