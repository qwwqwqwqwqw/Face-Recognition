# coding:utf-8
import paddle.v2 as paddle

def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(), ch_in=None):
    """
    卷积 + BN层
    """
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, act=active_type)

def shortcut(ipt, ch_in, ch_out, stride):
    """
    残差连接
    """
    if ch_in != ch_out:
        return conv_bn_layer(ipt, ch_out, 1, stride, 0, paddle.activation.Linear())
    else:
        return ipt

def basicblock(ipt, ch_in, ch_out, stride):
    """
    基本残差块
    """
    tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Relu())

def layer_warp(block_func, ipt, ch_in, ch_out, count, stride):
    """
    堆叠残差块
    """
    tmp = block_func(ipt, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp

def resnet_face(ipt, class_dim):
    """
    人脸识别的ResNet模型
    """
    # 设置卷积组的深度
    depth = 20
    # 计算每组包含多少个卷积层
    n = (depth - 2) // 6
    # 设置特征图的数量
    nf = 32
    
    # 对输入图像进行预处理
    ipt_bn = ipt - 128.0
    
    # 第一个卷积层
    conv1 = conv_bn_layer(ipt_bn, ch_in=3, ch_out=nf, filter_size=3, stride=1, padding=1)
    
    # 残差块组
    conv2 = layer_warp(basicblock, conv1, nf, nf*2, n, 2)
    conv3 = layer_warp(basicblock, conv2, nf*2, nf*4, n, 2)
    conv4 = layer_warp(basicblock, conv3, nf*4, nf*8, n, 2)
    
    # 全局平均池化
    pool = paddle.layer.img_pool(
        input=conv4, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())
    
    # 特征层 - 用于人脸对比
    feature = paddle.layer.fc(
        input=pool, size=512, act=paddle.activation.Linear(),
        param_attr=paddle.attr.Param(initial_std=0.001))
    
    # 分类层 - 用于人脸识别
    output = paddle.layer.fc(
        input=feature, size=class_dim, act=paddle.activation.Softmax())
    
    return feature, output 