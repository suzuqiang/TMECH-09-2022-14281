import matplotlib.pyplot as plt
import numpy as np
#data是acc中三个箱型图的参数

data = [ [44.25, 33.42, 37.59, 35.79, 42.82 ],
         [44.48, 36.46, 40.86, 36.26, 44.78 ],
         [48.02, 47.86, 45.78, 45.31, 45.05 ],
         [72.86, 53.34, 62.49, 62.87, 55.73 ],
         [70.32, 62.14, 58.99, 68.84, 57.98 ],
         [77.99, 62.25, 59.62, 66.76, 56.51 ],
         [87.87, 67.64, 62.91, 70.41, 61.20 ]]
#data2 是F1 score中三个箱型图的参数
data2 = [[47.37, 46.58, 49.42, 50.67, 47.54 ],
         [46.15, 49.58, 54.47, 50.54, 52.62 ],
         [53.17, 46.55, 47.07, 55.54, 52.52 ],
         [75.73, 70.34, 57.48, 64.16, 59.43 ],
         [81.54, 71.69, 54.24, 55.86, 72.67 ],
         [88.61, 69.93, 64.09, 69.65, 70.87 ],
         [91.17, 72.24, 68.60, 70.82, 74.47 ]]
	#data3 是IoU中三个箱型图的参数
data3 = [[56.54, 53.79, 55.73, 51.48, 50.89 ],
         [55.67, 54.93, 59.32, 54.75, 50.31 ],
         [62.34, 58.21, 58.44, 58.37, 58.75 ],
         [68.25, 64.08, 70.54, 76.02, 66.94 ],
         [73.44, 80.64, 80.57, 77.46, 89.72 ],
         [87.82, 85.62, 84.41, 81.97, 89.41 ],
         [88.25, 89.75, 88.33, 85.00, 90.00 ]]

data4 = [[53.23, 58.26, 59.98, 53.59, 53.19 ],
         [55.42, 57.56, 63.23, 56.71, 56.19 ],
         [60.42, 65.26, 70.75, 62.39, 57.24 ],
         [84.58, 74.31, 88.52, 88.90, 89.33 ],
         [92.25, 79.45, 92.73, 86.31, 85.98 ],
         [88.93, 79.82, 92.58, 88.08, 87.84 ],
         [92.27, 85.70, 92.86, 89.42, 90.47 ]]
	#箱型图名称
labels = ["CNN", "SN", "PN","LST","TPN","RRPN","SSWCPN"]
fontsize = 10
widt = 0.25
sta = 0
wid = 2
wi = 0.225
x_position=[sta,sta+wid,sta+2*wid,sta+3*wid]
#三个箱型图的颜色 RGB （均为0~1的数据）
colors = [(202/255.,96/255.,17/255.), (255/255.,217/255.,102/255.),(137/255.,128/255.,68/255.),(65/255.,105/255.,225/255.),
          (205/255.,92/255.,92/255.), (255/255.,165 /255.,0/255.),(218/255.,112/255.,214/255.),(0 /255.,139 /255.,69/255.)]
	#绘制箱型图
	#patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3 
bplot = plt.boxplot(data,patch_artist=True,labels=labels,
                    medianprops={ 'lw':1,'ls':'-', 'color' : 'black' },
                    positions=(x_position[0],x_position[0]+widt,x_position[0]+2*widt,
                               x_position[0]+3*widt,x_position[0]+4*widt,x_position[0]+5*widt,x_position[0]+6*widt),
                    widths=wi,showfliers= True,
                    flierprops={ 'marker':'+','markersize':10},
                    ) 
	#将三个箱分别上色
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


bplot2 = plt.boxplot(data2,patch_artist=True,labels=labels,
                     medianprops={ 'lw':1,'ls':'-', 'color' : 'black' },
                     positions=(x_position[1],x_position[1]+widt,x_position[1]+2*widt,
                               x_position[1]+3*widt,x_position[1]+4*widt,x_position[1]+5*widt,x_position[1]+6*widt),
                     widths=wi,showfliers= True,
                     flierprops={ 'marker':'+','markersize':10},
                     ) 

for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)

bplot3 = plt.boxplot(data3, patch_artist=True,labels=labels,
                     medianprops={ 'lw':1,'ls':'-', 'color' : 'black' },
                     positions=(x_position[2],x_position[2]+widt,x_position[2]+2*widt,
                               x_position[2]+3*widt,x_position[2]+4*widt,x_position[2]+5*widt,x_position[2]+6*widt),
                     widths=wi,showfliers= True,
                     flierprops={ 'marker':'+','markersize':10},
                     )  

for patch, color in zip(bplot3['boxes'], colors):
    patch.set_facecolor(color)

bplot4 = plt.boxplot(data4, patch_artist=True,labels=labels,
                     medianprops={ 'lw':1,'ls':'-', 'color' : 'black' },
                     positions=(x_position[3],x_position[3]+widt,x_position[3]+2*widt,
                               x_position[3]+3*widt,x_position[3]+4*widt,x_position[3]+5*widt,x_position[3]+6*widt),
                     widths=wi,showfliers= True,
                     flierprops={ 'marker':'+','markersize':10},
                     )  

for patch, color in zip(bplot4['boxes'], colors):
    patch.set_facecolor(color)

x_position_fmt=["Trial A","Trial B","Trial C","Trial D"]
plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)
plt.ylim(20,100)
plt.ylabel('Accuarcy (%)',fontsize=fontsize),plt.xlabel('Item',fontsize=fontsize)
plt.grid(linestyle="-", alpha=0)  #绘制图中虚线 透明度0.3
plt.legend(bplot['boxes'],labels,loc='lower right',fontsize=fontsize)  #绘制表示框，右下角绘制
# plt.savefig(fname="pic.png",figsize=[10,10])  
plt.yticks(range(25,101,25),fontsize=fontsize)
plt.show()
