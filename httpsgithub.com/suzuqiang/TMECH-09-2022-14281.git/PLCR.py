import os,torch,warnings,sys
import numpy as np
import matplotlib.pyplot as plt
global args
warnings.filterwarnings('ignore')
PLLTrainHistoryDrowName = \
[    'Correct Query','All Query','Query Loss',\
     'Correct Unlabeled','All Unlabeled','Unlabeled Loss','ALL PseudoLabel','Correct PseudoLabel',\
     'Correct Test','All Test']
PreHIS = np.load('SaveModel/ProNet)PreTrainHistory)RandonNum4)10)3)100)100.npy')
PLLHIS = np.load('SaveModel/ProNet)PLLTrainHistory)RandonNum4)10)3)100)100.npy')
a = np.concatenate((PreHIS[:,[0,1]],PLLHIS[:,[0,1]]),0)

c = np.concatenate((PreHIS[:,[3,4]],PLLHIS[:,[8,9]]),0)


plt.subplot(3,1,1),plt.plot(a),plt.ylim(0,np.max(PreHIS[:,[0,1]]+1))
plt.subplot(3,1,2),plt.plot(PLLHIS[:,[4,6,7]]),plt.ylim(0,np.max(PLLHIS[:,[4,6,7]]+100))
plt.subplot(3,1,3),plt.plot(c),plt.ylim(0,np.max(PLLHIS[:,[8,9]]+100))
plt.plot(100*PLLHIS[:,3]/PLLHIS[:,4],label = 'Correct PseudoLabel Rate')
plt.plot(100*PLLHIS[:,7]/PLLHIS[:,6],label = 'Correct Unlabeled Rate')
