import os,torch,warnings,sys
import numpy as np
from Run import *
import matplotlib.pyplot as plt
global args
warnings.filterwarnings('ignore')
SetSeed()
args = Parse()
# _,_,UnlabeledTrainDataSet,_,_ = GetDataSetFewShot(args)
# Model = ProNet()
# Model.cuda()
# Data,Label = np.expand_dims(np.float32(np.load(args.data)),1),np.load(args.label)
# TureLabel = torch.from_numpy(Label).long()
# ActuralLabel = torch.from_numpy(Label).long()
# LabelUpdate,LabelUpdateIndex = np.zeros((len(TureLabel),args.epoch,np.max(Label)+1)),np.zeros(len(TureLabel))
# for epoch in range(args.epoch):
#     for step, (unlabeledinputs,LabelIndex) in enumerate(UnlabeledTrainDataSet):
#         unlabeledinputs = unlabeledinputs.cuda()
#         unlabeledfeature = Model.blocks(unlabeledinputs)
#         New = Model.GetSoftMax(unlabeledfeature,LabelIndex.size()[0])
#         LabelIndex,New = LabelIndex.detach().numpy(),New.cpu().detach().numpy()
#         for i in range(len(LabelIndex)):
#             LabelUpdate[LabelIndex[i]][int(LabelUpdateIndex[LabelIndex[i]])] += New[i]   
#         LabelUpdateIndex[LabelIndex] += 1

import mpl_toolkits.axisartist as axisartist
from mpl_toolkits.mplot3d import Axes3D #画三维图不可少
from matplotlib import cm  #cm 是colormap的简写

LabelUpdate = np.load('LabelUpdate)10)3.npy')
LabelUpdate = LabelUpdate.transpose(0,2,1)
Unlabeledtrainlabel = np.load('Unlabeledtrainlabel.npy')
Querytrainlabel = np.load('Querytrainlabel.npy')

plt.figure(1)
for i in range(36):
    a = np.argmax(np.mean(LabelUpdate[Unlabeledtrainlabel[i]],1))
    b = Unlabeledtrainlabel[i] // 800
    plt.subplot(12,3,i+1),plt.imshow(LabelUpdate[Unlabeledtrainlabel[i]]),plt.xticks([]),plt.yticks([]),plt.title(a==b)

plt.figure(2)   
for i in range(45):
    a = np.argmax(np.mean(LabelUpdate[Unlabeledtrainlabel[i]],1))
    b = Unlabeledtrainlabel[i] // 800
    plt.subplot(15,3,i+1),plt.imshow(LabelUpdate[Querytrainlabel[i]]),plt.xticks([]),plt.yticks([]),plt.title(a==b)


