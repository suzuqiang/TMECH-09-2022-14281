import matplotlib.pyplot as plt
import numpy as np
#data是acc中三个箱型图的参数

data = [ [31.94, 37.00, 38.95, 35.42, 38.53 ],
         [44.42, 45.47, 51.78, 50.27, 52.96 ],
         [59.83, 55.63, 64.75, 62.53, 59.60 ],
         [61.75, 72.15, 70.98, 69.26, 68.86 ],
         [64.71, 72.78, 78.49, 73.29, 73.27 ],
         # [66.38, 68.94, 76.50, 84.94, 77.94 ],
         [79.19, 77.06, 81.19, 88.19, 83.25 ]]
#data2 是F1 score中三个箱型图的参数
data2 = [[49.06, 40.77, 49.06, 45.14, 46.31 ],
         [61.85, 54.26, 61.65, 60.23, 63.95 ],
         [62.83, 69.86, 72.81, 69.13, 66.93 ],
         [70.32, 74.08, 74.23, 77.04, 75.80 ],
         [79.32, 72.28, 76.46, 78.27, 74.81 ],
         # [77.20, 79.63, 87.50, 87.71, 84.03 ],
         [87.64, 94.06, 90.57, 92.99, 93.09 ]]
	#data3 是IoU中三个箱型图的参数
data3 = [[53.65, 50.14, 61.03, 59.78, 55.16 ],
         [66.38, 64.99, 65.50, 64.69, 69.25 ],
         [77.77, 80.32, 79.38, 79.36, 73.29 ],
         [83.79, 79.88, 85.43, 80.36, 83.72 ],
         [85.76, 83.96, 88.46, 94.60, 88.30 ],
         # [96.78, 85.03, 84.61, 92.14, 90.39 ],
         [96.35, 91.29, 94.61, 95.58, 94.83 ]]

data4 = [[60.19, 58.21, 70.21, 64.09, 59.64 ],
         [72.89, 74.03, 78.25, 78.56, 76.56 ],
         [83.03, 84.78, 81.56, 83.44, 80.19 ],
         [91.52, 85.07, 91.85, 90.14, 89.12 ],
         [93.55, 92.73, 90.63, 91.64, 92.88 ],
         # [93.00, 97.48, 94.86, 92.20, 96.22 ],
         [98.97, 95.96, 95.77, 96.56, 96.31 ]]
	#箱型图名称
labels = ["CNN", "SN", "PN","LST","TPN","SSWCPN"]
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
                               x_position[0]+3*widt,x_position[0]+4*widt,x_position[0]+5*widt),
                    widths=wi,showfliers= True,
                    flierprops={ 'marker':'+','markersize':10},
                    ) 
	#将三个箱分别上色
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


bplot2 = plt.boxplot(data2,patch_artist=True,labels=labels,
                     medianprops={ 'lw':1,'ls':'-', 'color' : 'black' },
                     positions=(x_position[1],x_position[1]+widt,x_position[1]+2*widt,
                               x_position[1]+3*widt,x_position[1]+4*widt,x_position[1]+5*widt),
                     widths=wi,showfliers= True,
                     flierprops={ 'marker':'+','markersize':10},
                     ) 

for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)

bplot3 = plt.boxplot(data3, patch_artist=True,labels=labels,
                     medianprops={ 'lw':1,'ls':'-', 'color' : 'black' },
                     positions=(x_position[2],x_position[2]+widt,x_position[2]+2*widt,
                               x_position[2]+3*widt,x_position[2]+4*widt,x_position[2]+5*widt),
                     widths=wi,showfliers= True,
                     flierprops={ 'marker':'+','markersize':10},
                     )  

for patch, color in zip(bplot3['boxes'], colors):
    patch.set_facecolor(color)

bplot4 = plt.boxplot(data4, patch_artist=True,labels=labels,
                     medianprops={ 'lw':1,'ls':'-', 'color' : 'black' },
                     positions=(x_position[3],x_position[3]+widt,x_position[3]+2*widt,
                               x_position[3]+3*widt,x_position[3]+4*widt,x_position[3]+5*widt),
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
