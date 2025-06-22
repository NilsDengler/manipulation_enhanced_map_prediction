"""
These scripts for creating a resnetUNet was borrowed from the mapping code in 
Georgakis, G., Bucher, B., Schmeckpeper, K., Singh, S., & Daniilidis, K. (2022). 
Learning to Map for Active Semantic Goal Navigation. International Conference on Learning Representations.
https://github.com/ggeorgak11/L2M/blob/master/models/networks/resnetUnet.py
and edited by us
"""

import torch
import torch.nn as nn
from torchvision import models
import lightning as L
import pdb 
import numpy as np


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_channel_in, n_class_out,epsilon = 0.1,do_dropout = False,max_alpha = 50,normalize = False):
        super().__init__()
        self.relu = nn.ReLU(inplace = True)
        self.base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.low_layer = nn.Sequential(convrelu(128,256,3,1),convrelu(256,512,3,1))
        # self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        # self.layer3_1x1 = convrelu(256, 256, 1, 0)
        # self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        # self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 256, 3, 1)

        self.occupancy_head = nn.Conv2d(256, n_class_out, 1)
        self.epsilon = epsilon
        self.do_dropout = do_dropout
        self.dropout = nn.Dropout(0.3,inplace = False)
        self.normalize = normalize
        self.max_alpha = max_alpha
        self.a = 2/(self.max_alpha-1)
        self.b = -1-self.a
        self.tanh = nn.Tanh()
        self.sigmoid =nn.Sigmoid
    def process_dropout_input(self,x):
        if(self.do_dropout):
            return self.dropout(x)
        else:
            return x
    def forward(self, input):

        B,C, cH, cW = input.shape
        # input = input.view(B*T,C,cH,cW)
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(self.process_dropout_input(input))
        layer1 = self.layer1(self.process_dropout_input(layer0))
        layer2 = self.layer2(self.process_dropout_input(layer1))
        # layer3 = self.layer3(layer2)
        # layer4 = self.layer4(layer3)

        # layer4 = self.layer4_1x1(layer4)
        # x = self.upsample(layer4)
        
        # layer3 = self.layer3_1x1(layer3)
        # x = torch.cat([x, layer3], dim=1)
        # x = self.conv_up3(x)

        x = self.low_layer(self.process_dropout_input(layer2))
        layer2 = self.layer2_1x1(self.process_dropout_input(layer2))
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(self.process_dropout_input(x))

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(self.process_dropout_input(x))

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(self.process_dropout_input(x))

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(self.process_dropout_input(x))

        occupancy = self.occupancy_head(x)
        # if(self.evidential):
        #     occupancy2 = self.relu(occupancy) + 1 + self.epsilon
        # else:
        #     occupancy2 = occupancy
        return occupancy
    
    # def get_occupancy_2D(self,occupancy2):
    #     alpha = occupancy2[:,1::2,:,:]
    #     beta = occupancy2[:,::2,:,:]
    #     alpha_zero = alpha+beta
    #     p = alpha/(alpha_zero)
    #     alpha_zero_2D = alpha_zero[:,10:20,].max(axis = 1)[0]
    #     if(self.normalize):
    #         occupancy_2D = self.normalize_distributions(occupancy_2D)
    #     occupancy_2D = torch.stack([beta_2d,alpha_2d],dim = 1)

    #     return occupancy_2D
    
    def get_occupancy_2D(self,occupancy2):
        alpha = occupancy2[:,1::2,:,:]
        beta = occupancy2[:,::2,:,:]
        # occupancy_beta = torch.stack([beta,alpha],dim = 1)
        # occupancy_beta = occupancy_beta/occupancy_beta.sum(axis =1,keepdims = True)
        # max_alpha = torch.argmax(alpha[:,10:],axis = 1)
        # if((self.idxs is None) or self.idxs.shape != (max_alpha.shape)):
        #     self.idxs = np.indices(max_alpha.shape)
        # alpha_2d = alpha[self.idxs[0,:],max_alpha,self.idxs[1,:],self.idxs[2,:]]
        alpha_2d = alpha[:,10,:,:]
        beta_2d = beta[:,10,:,:]
        occupancy_2D = torch.stack([beta_2d,alpha_2d],dim = 1)
        if(self.normalize):
            occupancy_2D = torch.clamp(occupancy_2D,1+self.epsilon,self.max_alpha)
            occupancy_2D = occupancy_2D*self.a + self.b
        del alpha_2d,beta_2d

        return occupancy_2D

    def normalize_distributions(self,distribution):
        distribution = torch.clamp(distribution,1+self.epsilon,self.max_alpha)
        return distribution*self.a + self.b
        # return self.tanh(distribution)
    
    def denormalize_distributions(self,distribution):
        return self.relu(distribution) + 1 + self.epsilon#(self.tanh(distribution)-self.b)/self.a + self.epsilon

class SemanticUNet(UNet):
    def __init__(self, n_channel_in, n_class_out,epsilon = 0.001,n_classes = 7,n_semantic_channel_in = 7,do_dropout = False,max_alpha = 50,normalize = False):
        super().__init__(n_channel_in, n_class_out,epsilon = epsilon,do_dropout=do_dropout,max_alpha = max_alpha,normalize = normalize)
        self.n_classes = n_classes
        self.n_semantic_channel_in = n_semantic_channel_in
        self.max_alpha = max_alpha
        self.normalize = normalize
        self.semantic_head = UNet(2+self.n_semantic_channel_in,n_class_out = self.n_classes,epsilon = epsilon,do_dropout = do_dropout,max_alpha = max_alpha)
        self.idxs = None
        self.a = 2/(self.max_alpha-1)
        self.b = -1-self.a
        self.idxs = None

    def forward(self, occupancy_features,semantic_features):
        B,C, cH, cW = occupancy_features.shape
        # input = input.view(B*T,C,cH,cW)
        x_original = self.conv_original_size0(occupancy_features)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(self.process_dropout_input(occupancy_features))
        layer1 = self.layer1(self.process_dropout_input(layer0))
        layer2 = self.layer2(self.process_dropout_input(layer1))
        # layer3 = self.layer3(layer2)
        # layer4 = self.layer4(layer3)

        # layer4 = self.layer4_1x1(layer4)
        # x = self.upsample(layer4)
        
        # layer3 = self.layer3_1x1(layer3)
        # x = torch.cat([x, layer3], dim=1)
        # x = self.conv_up3(x)

        x = self.low_layer(self.process_dropout_input(layer2))
        layer2 = self.layer2_1x1(self.process_dropout_input(layer2))
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(self.process_dropout_input(x))

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(self.process_dropout_input(x))

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(self.process_dropout_input(x))

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(self.process_dropout_input(x))

        occupancy = self.occupancy_head(x)
        occupancy2 = self.denormalize_distributions(occupancy)
        alpha = occupancy2[:,1::2,:,:]
        beta = occupancy2[:,::2,:,:]
        alpha_zero = alpha+beta
        occupancy_prob = alpha/(alpha_zero)
        occupancy_prob_2d = occupancy_prob[:,15:16,:,:]
        if(self.normalize):
            alpha_zero = torch.clamp(alpha_zero,1+self.epsilon,self.max_alpha)
            occupancy_alpha_2d = alpha_zero[:,15,:,:].unsqueeze(1)*self.a+self.b
        else:
            occupancy_alpha_2d = alpha_zero[:,15:16,:,:]
        # import cv2
        # import numpy as np
        # tmp = occupancy_prob_2d.detach().cpu().numpy()

        # for i in range(10,60,5):
        #     cv2.imshow('occupancy_2d_{}'.format(i),occupancy_prob.detach().cpu().numpy()[0,i,:,:])

        occupancy_input = torch.cat([occupancy_prob_2d,occupancy_alpha_2d],dim = 1)
        # occupancy_2D = self.get_occupancy_2D(occupancy2)
        # occupancy_2D = torch.max(occupancy_beta[:,:,10:,:,:],dim = 2,keepdim = False)[0]
        x = torch.cat([occupancy_input,semantic_features],dim = 1)
        semantics = self.semantic_head(x)

        semantics = self.denormalize_distributions(semantics)
        # my_cmap = get_my_cmap(n_classes = 15)
        # tmp2 = semantics.detach().cpu().numpy()
        # tmp3 = tmp2.argmax(axis =1)[0]
        # cv2.imshow('semantics',my_cmap[tmp3])
        # key = cv2.waitKey(10)
        # if((key&0xFF) == ord('d')):
        #     import pdb
        #     import matplotlib 
        #     matplotlib.use('TkAgg')
        #     from matplotlib import pyplot as plt
        #     pdb.set_trace()

        return occupancy2,semantics

    



class PushSemanticUNet(SemanticUNet):
    def __init__(self, n_channel_in, n_class_out,epsilon = 0.001,n_classes = 7,n_semantic_channel_in = 7,do_dropout = False,max_alpha = 50,normalize = False):
        super().__init__(n_channel_in, n_class_out,epsilon = epsilon,do_dropout=do_dropout,max_alpha = max_alpha,normalize = normalize)
        self.n_classes = n_classes
        self.n_semantic_channel_in = n_semantic_channel_in
        self.semantic_head = UNet(4+self.n_semantic_channel_in,n_class_out = n_classes,epsilon = 0.001,do_dropout = do_dropout,max_alpha = max_alpha)
        self.difference_head = nn.Sequential(convrelu(256,64, 3, 1),convrelu(64,32,3, 1),nn.Conv2d(32, 2, 1))
        self.max_alpha = max_alpha
        self.a = 2/(self.max_alpha-1)
        self.b = -1-self.a
        self.idxs = None
        self.sigmoid = nn.Sigmoid()
    def forward(self, occupancy_features,semantic_features):

        B,C, cH, cW = occupancy_features.shape
        # input = input.view(B*T,C,cH,cW)
        x_original = self.conv_original_size0(occupancy_features)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(self.process_dropout_input(occupancy_features))
        layer1 = self.layer1(self.process_dropout_input(layer0))
        layer2 = self.layer2(self.process_dropout_input(layer1))

        x = self.low_layer(self.process_dropout_input(layer2))
        layer2 = self.layer2_1x1(self.process_dropout_input(layer2))
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(self.process_dropout_input(x))

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(self.process_dropout_input(x))

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(self.process_dropout_input(x))

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(self.process_dropout_input(x))
        difference = self.difference_head(x)
        difference = self.denormalize_distributions(difference)#self.relu(self.difference_head(x))+1+self.epsilon
        occupancy = self.occupancy_head(x)

        occupancy2 = self.denormalize_distributions(occupancy)#self.relu(occupancy) + 1 + self.epsilon
        
        alpha = occupancy2[:,1::2,:,:]
        beta = occupancy2[:,::2,:,:]
        # occupancy_beta = torch.stack([beta,alpha],dim = 1)
        # occupancy_beta = occupancy_beta/occupancy_beta.sum(axis =1,keepdims = True)
        occupancy_2D = self.get_occupancy_2D(occupancy2)

        occupancy_beta = torch.stack([beta,alpha],dim = 1)
        if(self.normalize):
            d1 = torch.clamp(difference,1+self.epsilon,self.max_alpha)
            difference_renorm = self.normalize_distributions(d1)
            # occupancy_2D = occupancy_2D
        else:
            difference_renorm = difference
        x = torch.cat([occupancy_2D,difference_renorm,semantic_features],dim = 1)
        semantics = self.semantic_head(x)


        # semantics = self.relu(semantics) + 1 + self.epsilon
        semantics = self.denormalize_distributions(semantics)


        return occupancy_beta,semantics,difference,occupancy2
    def denormalize_distributions(self,distribution):
        return self.relu(distribution) + 1 + self.epsilon#self.max_alpha*self.sigmoid(distribution) + 1 + self.epsilon



class NonEvidentialSemanticUNet(SemanticUNet):
    def __init__(self, n_channel_in, n_class_out,epsilon = 0.001,n_classes = 7,n_semantic_channel_in = 7,do_dropout = False,max_alpha = 50):
        super().__init__(n_channel_in, n_class_out,epsilon,n_classes,n_semantic_channel_in,do_dropout,max_alpha)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.semantic_head = UNet(102+self.n_semantic_channel_in,n_class_out = n_classes,epsilon = 0.001,do_dropout = do_dropout,max_alpha = max_alpha)

    def forward(self, occupancy_features,semantic_features):
        B,C, cH, cW = occupancy_features.shape
        # input = input.view(B*T,C,cH,cW)
        x_original = self.conv_original_size0(occupancy_features)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(self.process_dropout_input(occupancy_features))
        layer1 = self.layer1(self.process_dropout_input(layer0))
        layer2 = self.layer2(self.process_dropout_input(layer1))

        x = self.low_layer(self.process_dropout_input(layer2))
        layer2 = self.layer2_1x1(self.process_dropout_input(layer2))
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(self.process_dropout_input(x))

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(self.process_dropout_input(x))

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(self.process_dropout_input(x))

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(self.process_dropout_input(x))
        occupancy = self.occupancy_head(x)
        occupancy = self.sigmoid(occupancy)
        # occupancy_beta = torch.stack([beta,alpha],dim = 1)
        # occupancy_beta = occupancy_beta/occupancy_beta.sum(axis =1,keepdims = True)
        # occupancy_2D = self.get_occupancy_2D(occupancy)

        x = torch.cat([occupancy,semantic_features],dim = 1)
        semantics = self.semantic_head(x)

        return occupancy,semantics

    def get_occupancy_2D(self,occupancy):
        occupancy_2D = torch.max(occupancy[:,10:,:,:],dim = 1,keepdim = True)[0]
        return occupancy_2D

    

class NonEvidentialPushSemanticUNet(NonEvidentialSemanticUNet):
    def __init__(self, n_channel_in, n_class_out,epsilon = 0.001,n_classes = 7,n_semantic_channel_in = 7,do_dropout = False,max_alpha = 50,normalize = False):
        super().__init__(n_channel_in, n_class_out,epsilon = epsilon,do_dropout=do_dropout,max_alpha = max_alpha,normalize = normalize)
        self.n_classes = n_classes
        self.n_semantic_channel_in = n_semantic_channel_in
        self.semantic_head = UNet(4+self.n_semantic_channel_in,n_class_out = n_classes,epsilon = 0.001,do_dropout = do_dropout,max_alpha = max_alpha)
        self.difference_head = nn.Sequential(convrelu(256,64, 3, 1),convrelu(64,32,3, 1),nn.Conv2d(32, 1, 1))
        self.max_alpha = max_alpha
        self.a = 2/(self.max_alpha-1)
        self.b = -1-self.a
        self.idxs = None
        self.sigmoid = nn.Sigmoid()
    def forward(self, occupancy_features,semantic_features):

        B,C, cH, cW = occupancy_features.shape
        # input = input.view(B*T,C,cH,cW)
        x_original = self.conv_original_size0(occupancy_features)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(self.process_dropout_input(occupancy_features))
        layer1 = self.layer1(self.process_dropout_input(layer0))
        layer2 = self.layer2(self.process_dropout_input(layer1))

        x = self.low_layer(self.process_dropout_input(layer2))
        layer2 = self.layer2_1x1(self.process_dropout_input(layer2))
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(self.process_dropout_input(x))

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(self.process_dropout_input(x))

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(self.process_dropout_input(x))

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(self.process_dropout_input(x))
        difference = self.difference_head(x)
        difference = self.denormalize_distributions(difference)#self.relu(self.difference_head(x))+1+self.epsilon
        occupancy = self.occupancy_head(x)

        occupancy2 = self.denormalize_distributions(occupancy)#self.relu(occupancy) + 1 + self.epsilon
        
        alpha = occupancy2[:,1::2,:,:]
        beta = occupancy2[:,::2,:,:]
        # occupancy_beta = torch.stack([beta,alpha],dim = 1)
        # occupancy_beta = occupancy_beta/occupancy_beta.sum(axis =1,keepdims = True)
        occupancy_2D = self.get_occupancy_2D(occupancy2)

        occupancy_beta = torch.stack([beta,alpha],dim = 1)
        if(self.normalize):
            d1 = torch.clamp(difference,1+self.epsilon,self.max_alpha)
            difference_renorm = self.normalize_distributions(d1)
            # occupancy_2D = occupancy_2D
        else:
            difference_renorm = difference
        x = torch.cat([occupancy_2D,difference_renorm,semantic_features],dim = 1)
        semantics = self.semantic_head(x)


        # semantics = self.relu(semantics) + 1 + self.epsilon
        semantics = self.denormalize_distributions(semantics)


        return occupancy_beta,semantics,difference,occupancy2
    def denormalize_distributions(self,distribution):
        return self.relu(distribution) + 1 + self.epsilon#self.max_alpha*self.sigmoid(distribution) + 1 + self.epsilon