import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from mpl_toolkits.mplot3d import Axes3D #画三维图不可少
from matplotlib import cm  #cm 是colormap的简写

# 1_dimension gaussian function
def gaussian(x,mu,sigma):
    f_x = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-np.power(x-mu, 2.)/(2*np.power(sigma,2.)))
    return(f_x)

# 2_dimension gaussian function
def gaussian_2(x,y,mu_x,mu_y,sigma_x,sigma_y):
    f_x_y = 1/(sigma_x*sigma_y*(np.sqrt(2*np.pi))**2)*np.exp(-np.power\
              (x-mu_x, 2.)/(2*np.power(sigma_x,2.))-np.power(y-mu_y, 2.)/\
              (2*np.power(sigma_y,2.)))
    return(f_x_y)

#设置2维表格
    

x_values = np.linspace(0,100,100)
y_values = np.linspace(0,8,8)
X,Y = np.meshgrid(x_values,y_values)
#高斯函数
mu_x,mu_y,sigma_x,sigma_y = 0,0,0.8,0.8
F_x_y = gaussian_2(X,Y,mu_x,mu_y,sigma_x,sigma_y)
#显示三维图
fig = plt.figure()
ax = plt.gca(projection='3d')
ax.plot_surface(X,Y,F_x_y,cmap='jet')
# 显示等高线图
#ax.contour3D(X,Y,F_x_y,50,cmap='jet')

