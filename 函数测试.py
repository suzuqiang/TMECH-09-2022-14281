import os,torch,warnings,sys
import numpy as np
from Schrobine import *
import matplotlib.pyplot as plt
global args
warnings.filterwarnings('ignore')
SetSeed()
args = Parse()
labelupdateindex=100
TureLabel = torch.from_numpy(np.load('Label1024.npy')).long()
ActuralLabel = torch.from_numpy(np.load('Label1024.npy')).long()
LabelUpdate = np.load('LabelUpdate4)25)16)38.npy') #6400*100*8
Unlabeledtrainlabel = np.load('Unlabeledtrainlabel.npy')
LabelIndex = Unlabeledtrainlabel[:48]
TempLabelUpdate = LabelUpdate[LabelIndex][:,labelupdateindex-args.length:labelupdateindex]
selected,pseudolabel,test = [],[],[]
for i in range(len(LabelIndex)):
    n = np.max(np.bincount(np.argmax(TempLabelUpdate[i],1)))
    if np.max(np.bincount(np.argmax(TempLabelUpdate[i],1))) > 0.75*args.length:
        pseudolabel.append(np.where(np.bincount(np.argmax(TempLabelUpdate[i],1))==n)[0][0])
        selected.append(LabelIndex[i])
        test.append(i)
        pseudolabel = np.array(pseudolabel)
for i in range(10):
    plt.subplot(5,2,i+1),plt.imshow(np.transpose(TempLabelUpdate[test[i]],(1,0)))
    plt.title(str(pseudolabel[i][0])+'-'+str(TureLabel[selected[i]]))
    unlabeledlogits = LabelUpdate[:16,99]