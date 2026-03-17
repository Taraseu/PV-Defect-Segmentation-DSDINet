# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x

# ------------------------------------------------ new block----------------------------------------------------------------------------------

class CBR(nn.Layer):
    def __init__(self, in_c, out_c, kernel_size =3, padding = 1, dilation=1, act=True):
        super().__init__()
        self.conv = nn.Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2D(out_c)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = F.relu(x)
        return x
    

###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Layer):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = paddle.create_parameter(shape=[1], 
                                                    dtype='float32', 
                                                    default_initializer=paddle.nn.initializer.Constant(value=1.0))
        #self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.shape
        #proj_query = x.view(m_batchsize, C, -1)                  # B, C, (H*W)
        proj_query = x.reshape([m_batchsize, C, -1])  
    
        proj_key = x.reshape([m_batchsize, C, -1]).transpose([0, 2, 1])  # B, (H*W), C
        energy = paddle.bmm(proj_query, proj_key)            # 批量矩阵乘法，计算注意力得分  B, C, C
        #attention = self.softmax(energy)                    # (B, C, C)
        attention = F.softmax(energy, axis=-1)
        proj_value = x.reshape([m_batchsize, C, -1])             # B, C, (H*W)

        out = paddle.bmm(attention, proj_value)              #(B, C, (H*W))
        out = out.reshape([m_batchsize, C, height, width])
        #out = out.view(m_batchsize, C, height, width)       # B, C, H, W

        out = self.gamma * out + x
        return out
    

###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Layer):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = paddle.create_parameter(shape=[1], 
                                                    dtype='float32', 
                                                    default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.shape
        proj_query = self.query_conv(x).reshape([m_batchsize, -1, width * height]).transpose([0, 2, 1])   # ( B, (H*W), C//8)
        proj_key = self.key_conv(x).reshape([m_batchsize, -1, width * height])                        # ( B, C//8, (H*W))
        energy = paddle.bmm(proj_query, proj_key)                                                 #(B, H*W, H*W)
        attention = F.softmax(energy)         
        proj_value = self.value_conv(x).reshape([m_batchsize, -1, width * height])                    # (B, C, (H*W))

        out = paddle.bmm(proj_value, attention.transpose([0, 2, 1]))                                    # (B, C, （H*W))
        out = out.reshape([m_batchsize, C, height, width])

        out = self.gamma * out + x
        return out

###################################################################
# ##################### Attention Module ########################
###################################################################
class CASA(nn.Layer):
    def __init__(self, channel):
        super(CASA, self).__init__()
        self.channel = channel
        self.ca = CA_Block(self.channel)
        self.sa = SA_Block(self.channel)
        
        self.conv1 = CBR(self.channel * 2, self.channel, kernel_size=1, padding=0)
        self.conv2 = CBR(self.channel, self.channel, act=False)
        self.conv3 = CBR(self.channel, self.channel, act=False)
        self.conv4 = CBR(self.channel, self.channel, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU()
        #self.map = nn.Conv2D(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        x0, x1 = paddle.chunk(x, 2, axis=2) # x0: [B, C, H/2, W], x1: [B, C, H/2, W]
        x0 = paddle.chunk(x0, 2, axis=3)           # x0: [B, C, H/2, W/2]
        x1 = paddle.chunk(x1, 2, axis=3)           # x1: [B, C, H/2, W/2]    x1 也是一个包含两个张量的列表
        
    
        x0_c = [self.ca(x0[-2]) , self.ca(x0[-1]) ]
        x0_s = [self.sa(x0[-2]), self.sa(x0[-1])]
        x0_fusion = [x0_c[i] + x0_s[i] for i in range(len(x0_c))]  # 使用列表生成式合并
        
        x1_c = [self.ca(x1[-2]), self.ca(x1[-1])]
        x1_s = [self.sa(x1[-2]), self.sa(x1[-1])]
        x1_fusion = [x1_c[i] + x1_s[i] for i in range(len(x1_c))]
        
        x0_attention = paddle.concat(x0_fusion, axis=3)
        x1_attention = paddle.concat(x1_fusion, axis=3)
        
        x3 = paddle.concat((x0_attention, x1_attention), axis=2)
        
        x_a = self.ca(x)
        x_s = self.sa(x)       # feature map
        x_attention = x_a + x_s
        
        x4 = paddle.concat([x3, x_attention], axis=1)
        x4 = self.conv1(x4)

        s1 = x4
        x4 = self.conv2(x4)
        x4 = self.relu(x4 + s1)

        s2 = x4
        x4 = self.conv3(x4)
        x4 = self.relu(x4 + s2 + s1)

        s3 = x4
        x4 = self.conv4(x4)
        x4 = self.relu(x4 + s3 + s2 + s1)
        
        map = x_attention + x3      # 原先是两个attention map相加
        #map = self.map(sab)        # attention map    

        return x4

###################################################################
# ################## Context Exploration Block ####################
###################################################################

class Context_Exploration(nn.Layer):
    def __init__(self, input_channels):
        super(Context_Exploration, self).__init__()
        self.input_channels = input_channels

        self.pre = nn.Sequential(
            nn.Conv2D(self.input_channels, self.input_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2D(self.input_channels), nn.ReLU())


        self.p1_dc = nn.Sequential(
            nn.Conv2D(self.input_channels, self.input_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2D(self.input_channels), nn.ReLU())

        self.p2_dc = nn.Sequential(
            nn.Conv2D(self.input_channels, self.input_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2D(self.input_channels), nn.ReLU())

        self.p3_dc = nn.Sequential(
            nn.Conv2D(self.input_channels, self.input_channels, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm2D(self.input_channels), nn.ReLU())

        #self.p4_dc = nn.Sequential(
        #    nn.Conv2D(self.input_channels, self.input_channels, kernel_size=3, padding=4, dilation=2),
        #    nn.BatchNorm2D(self.input_channels), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2D(self.input_channels * 4, self.input_channels, kernel_size=1, stride=1, padding=0, dilation=1),
                                    nn.BatchNorm2D(self.input_channels), nn.ReLU())

        self.post = nn.Sequential(
            nn.Conv2D(self.input_channels, self.input_channels, 1, padding=0, dilation=1),
            nn.BatchNorm2D(self.input_channels), nn.ReLU())
    def forward(self, x):
        
        y0 = self.pre(x)
        y1 = self.p1_dc(x+y0)
        y2 = self.p2_dc(x+y1)
        y3 = self.p3_dc(y2+x)

        # 修改拼接操作以包含 p3_dc 和 p4_dc
        ce = paddle.concat([y0, y1, y2, y3], axis=1)
        ce = self.fusion(ce)
        #ce = self.post(ce + x)
        return ce + x

#

# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Layer):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1         # current-level features
        self.channel2 = channel2         # higher-level features

        self.up = nn.Sequential(nn.Conv2D(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2D(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2D(scale_factor=2))
        
        self.mix = nn.Sequential(nn.Conv2D(channel1 * 3, channel1 * 2, kernel_size=3, padding=1),nn.BatchNorm2D(channel1 * 2), nn.ReLU(),nn.Conv2D(channel1 * 2, channel1, kernel_size=3, padding=1))   
        
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2D(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2D(self.channel1, 1, 7, 1, 3)   # 只适用于二值分割

        #self.fp = Context_Exploration_Block(self.channel1)
        #self.fn = Context_Exploration_Block(self.channel1)    #remove the CE block

        self.alpha = paddle.create_parameter(shape=[1], 
                                                    dtype='float32', 
                                                    default_initializer=paddle.nn.initializer.Constant(value=1.0))
        self.beta = paddle.create_parameter(shape=[1], 
                                                    dtype='float32', 
                                                    default_initializer=paddle.nn.initializer.Constant(value=1.0))
        
        self.gamma = paddle.create_parameter(shape=[1], 
                                                    dtype='float32', 
                                                    default_initializer=paddle.nn.initializer.Constant(value=1.0))
        self.bn1 = nn.BatchNorm2D(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2D(self.channel1)
        self.relu2 = nn.ReLU()

        self.ce_f = Context_Exploration(self.channel1)
        #self.ce_b = Context_Exploration(self.channel1)
        #self.ce_x = Context_Exploration(self.channel1)
        self.convb = nn.Sequential(nn.Conv2D(self.channel1, self.channel1, 3, 1, 1), nn.BatchNorm2D(self.channel1), nn.ReLU())
        self.convx = nn.Sequential(nn.Conv2D(self.channel1, self.channel1, 3, 1, 1), nn.BatchNorm2D(self.channel1), nn.ReLU())

    def forward(self, x, y, in_map):
        # x; current-level features   (B, C, H, W)
        # y: higher-level features 
        # in_map: higher-level prediction

        up = self.up(y)  

        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        f_ce = self.ce_f(f_feature)
        b_feature = self.convb(b_feature) 
        x = self.convx(x)
        
        combined = paddle.concat([self.alpha * f_ce, self.beta * b_feature, self.gamma * x], axis=1)    #(B, 4C, H, W) 
        combined = self.mix(combined)
        refine = combined + up

        output_map = self.output_map(refine)

        return refine, output_map

#------------------------------------------------------------------------------------------------------------------------------



@manager.MODELS.add_component     
class ELFormer(nn.Layer):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        #super(ELFormer, self).__init__()  一种兼容 Python 2 和 3 的写法
        super().__init__()


        
        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(in_channels=embedding_dim * 4,
                                             out_channels=embedding_dim,
                                             kernel_size=1,
                                             bias_attr=False)

        self.linear_pred = nn.Conv2D(embedding_dim,
                                     self.num_classes,
                                     kernel_size=1)
        
        #---------------------------------------------new block-------------------------------
        
        self.cr1 = nn.Sequential(nn.Conv2D(c1_in_channels, c1_in_channels, 3, 1, 1), nn.BatchNorm2D(c1_in_channels), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2D(c2_in_channels, c2_in_channels, 3, 1, 1), nn.BatchNorm2D(c2_in_channels), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2D(c3_in_channels, c3_in_channels, 3, 1, 1), nn.BatchNorm2D(c3_in_channels), nn.ReLU())
        self.cr4 = nn.Sequential(nn.Conv2D(c4_in_channels, c4_in_channels, 3, 1, 1), nn.BatchNorm2D(c4_in_channels), nn.ReLU())

        self.focus1 = Focus(c1_in_channels, c2_in_channels)
        self.focus2 = Focus(c2_in_channels, c3_in_channels)
        self.focus3 = Focus(c3_in_channels, c4_in_channels)

        #self.atten1 = CASA(c1_in_channels)
        #self.atten2 = CASA(c2_in_channels)       
        #self.atten3 = CASA(c3_in_channels)
        self.atten4 = CASA(c4_in_channels)

        self.up_c4_to_c3 = nn.Conv2D(c4_in_channels, c3_in_channels, kernel_size=1, stride=1, padding=0)
        self.up_c4_to_c2 = nn.Conv2D(c4_in_channels, c2_in_channels, kernel_size=1, stride=1, padding=0)
        self.up_c4_to_c1 = nn.Conv2D(c4_in_channels, c1_in_channels, kernel_size=1, stride=1, padding=0)


        self.pre4 = nn.Conv2D(c4_in_channels, 1, 7, 1, 3)       # 只能用于二值分割

        #----------------------------------------------new block------------------------------


        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats      

        ############## MLP decoder on C1-C4 ###########
        c1_shape = c1.shape   # [B, C1, H/4, W/4]      [4, 64, 128, 128]  batch_size=4 image_size=512x512
        c2_shape = c2.shape   # [B, C2, H/8, W/8]      [4, 128, 64, 64]
        c3_shape = c3.shape   # [B, C3, H/16, W/16]    [4, 320, 32, 32]
        c4_shape = c4.shape   # [B, C4, H/32, W/32]    [4, 512, 16, 16]

#---------------------------------------------new block-------------------------------

        ############## conv encoder on C1-C4 ###########
        cr1 = self.cr1(c1)    # channel not change
        cr2 = self.cr2(c2)
        cr3 = self.cr3(c3)
        cr4 = self.cr4(c4)


        ############## attention on C1-C4 ###########
        #atten1 = self.atten1(cr1)    # feature map 
        #atten2 = self.atten2(cr2)
        #atten3 = self.atten3(cr3)
        atten4 = self.atten4(cr4)     # [B, C4, H/32, W/32]   e4_refine

        c3_tensor = F.interpolate(self.up_c4_to_c3(c4), size=c3_shape[2:], mode='bilinear', align_corners=False)
        c2_tensor = F.interpolate(self.up_c4_to_c2(c4), size=c2_shape[2:], mode='bilinear', align_corners=False)
        c1_tensor = F.interpolate(self.up_c4_to_c1(c4), size=c1_shape[2:], mode='bilinear', align_corners=False)
        
        
        predict4 = self.pre4(atten4)                                  # channel not change 
        focus3, predict3 = self.focus3(cr3, atten4, predict4)  
        focus2, predict2 = self.focus2(cr2, focus3, predict3)
        focus1, predict1 = self.focus1(cr1, focus2, predict2)    


        ############## C1-C4统一大小 ###########
        _c4 = self.linear_c4(atten4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])            #_c4.shape [4, 768, 16, 16]

        _c4 = F.interpolate(_c4,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)  # [4, 768, 128, 128]

        _c3 = self.linear_c3(focus3 + c3_tensor).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(_c3,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)

        _c2 = self.linear_c2(focus2 + c2_tensor).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(_c2,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)

        _c1 = self.linear_c1(focus1 + c1_tensor).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        
        ############## concat and fuse ###########
        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return [
            F.interpolate(logit,
                          size=x.shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
        ]
