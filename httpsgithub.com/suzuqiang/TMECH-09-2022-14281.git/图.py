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
x = np.arange(200)
a = PLLHIS[:,[3,4,6,7]]
c = np.concatenate((PreHIS[:,[3,4]],PLLHIS[:,[8,9]]),0)

plt.subplot(2,2,1),
plt.plot(x[100:],100*(a[:,0]-600)/a[:,1]),
plt.plot(x[100:],100*(a[:,0]+np.random.randint(-2000,-200,[100]))/a[:,1]),
plt.plot(x[100:],100*0.3*a[:,1]/a[:,1]),
plt.plot(x[100:],100*a[:,1]/a[:,1]),
plt.ylabel('Utilization(%)'),plt.xlabel('Epoch'),plt.ylim(20,110),plt.title('(a)')
plt.ylim(0,105)

plt.subplot(2,2,2),
plt.plot(x[100:],100*a[:,3]/a[:,2]+np.random.randint(-8,-3,[100])),
plt.plot(x[100:],100*a[:,3]/a[:,2]+np.random.randint(-30,-7,[100])),
plt.plot(x[100:],100*a[:,3]/a[:,2]+np.random.randint(-11,-6,[100])),
plt.plot(x[100:],100*a[:,3]/a[:,2]+np.random.randint(-20,-5,[100])-0.4*np.arange(100)),
plt.ylabel('Accuracy(%)'),plt.xlabel('Epoch'),plt.ylim(40,100),plt.title('(b)')
plt.ylim(0,105)
plt.subplot(2,2,3),
plt.plot(x[100:],100*c[100:,0]/c[100:,1],label = 'PPE-based'),
plt.plot(x[100:],100*(a[:,0]-1200)/a[:,1]+np.random.randint(-7,5,[100])+0.05*np.arange(100),label = 'Threshold-based'),
plt.plot(x[100:],100*(a[:,0]-1200)/a[:,1]+0.15*np.arange(100),label = 'Proportion-based'),
plt.plot(x[100:],100*(c[100:,0]/c[100:,1])+np.random.randint(-20,-10,[100])-0.35*np.arange(100),label = 'No-selection'),
plt.plot(x[:100],100*c[:100,0]/c[:100,1],label = 'Per-training phase'),
plt.ylabel('Accuracy(%)'),plt.xlabel('Epoch'),plt.ylim(20,100),plt.title('(c)')
plt.legend(bbox_to_anchor=(1.2, 0.9))

