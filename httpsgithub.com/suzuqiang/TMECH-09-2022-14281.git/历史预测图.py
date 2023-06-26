import os,torch,warnings,sys
import numpy as np
from Schrobine import *
import matplotlib.pyplot as plt
global args
warnings.filterwarnings('ignore')
SetSeed()
args = Parse()
# LabelUpdate = np.load('LabelUpdate4)25)16)38.npy') #6400*100*8
# Querytrainlabel = np.load('Querytrainlabel.npy')

LabelUpdate = np.load('SaveModel/PreProNet)RandonNum0)10)3)100)100.npy') #6400*100*8
Querytrainlabel = np.load('Unlabeledtrainlabel.npy')
LabelUpdate = LabelUpdate[Querytrainlabel]
for i in range(0,20,1):
    plt.subplot(10,2,i+1),plt.imshow(LabelUpdate[7*i][:100,].transpose(1,0))
    plt.xticks([]),plt.yticks([])

ListNum=[0,2,3,4,8,10,9,17,24,29]
num = 0
for i in ListNum:
    plt.subplot(5,2,num+1),plt.imshow(LabelUpdate[i][:100,].transpose(1,0)),plt.title("("+str(num+1)+")",fontsize=25),
    plt.xticks([]),plt.yticks([])
    num +=1


for i in range(1,21,2):
    plt.subplot(10,2,(i//2)+1),plt.imshow(LabelUpdate[i][:100,].transpose(1,0)),plt.title("("+str((i//2)+1)+")",fontsize=25),
    plt.xticks([]),plt.yticks([])
plt.legend(fontsize=150)
plt.subplot(3,1,1),plt.imshow(LabelUpdate[25][:100,].transpose(1,0))
plt.subplot(3,1,3),plt.imshow(LabelUpdate[26][:100,].transpose(1,0))

plt.imshow(LabelUpdate[i][:100,].transpose(1,0)),plt.xticks([]),plt.yticks([])
plt.imshow(LabelUpdate[22][:100,].transpose(1,0)),plt.xticks([]),plt.yticks([])
test = np.load('E:/数据/DDSLabel_4_32_32.npy')
