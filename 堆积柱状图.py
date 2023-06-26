import matplotlib.pyplot as plt
import numpy as np
labels = np.arange(100)
men_means = np.arange(0,200,2)
women_means = np.arange(0,150,1.5)
Dog = np.arange(100)
Cat = 150*np.sin(labels)
width = 1.5       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots(figsize=(5,3),dpi=200)
ax.bar(labels, men_means, width,label='Men')
ax.bar(labels, women_means, width, bottom=men_means,label='Women')
ax.bar(labels, Dog, width, bottom=men_means,label='Dog')
ax.bar(labels, Cat, width, bottom=men_means,label='Cat')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.legend()
ax.text(.87,-.08,'\nVisualization by DataCharm',transform = ax.transAxes,
        ha='center', va='center',fontsize = 5,color='black',fontweight='bold',family='Roboto Mono')
# plt.savefig(r'F:\DataCharm\SCI paper plots\sci_bar_guanwang',width=5,height=3,
#             dpi=900,bbox_inches='tight')
plt.show()