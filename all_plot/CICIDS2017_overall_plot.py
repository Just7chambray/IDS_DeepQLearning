
# way 2

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick  


# whole = './CICIDS2017/TrafficLabelling_whole.csv'
# df = pd.read_csv(whole,low_memory=False)
# # print(df[' Label'].value_counts())
# df = df.replace('Web Attack � Brute Force','Web Attack - Brute Force')
# df = df.replace('Web Attack � XSS','Web Attack - XSS')
# df = df.replace('Web Attack � Sql Injection','Web Attack - Sql Injection')
# df.to_csv('./CICIDS2017/dataset_overall.csv')

path = './CICIDS2017/dataset_overall.csv'
# print(df[' Label'].value_counts())
# df = pd.read_csv(path,low_memory=False)
# x = df[' Label'].value_counts().index.tolist()
x = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest', 'Bot', 'Web Attack - Brute Force', 'Web Attack - XSS', 'Infiltration', 'Web Attack - Sql Injection', 'Heartbleed']
# y = df[' Label'].value_counts().values.tolist()
y = [2273097, 231073, 158930, 128027, 10293, 7938, 5897, 5796, 5499, 1966, 1507, 652, 36, 21, 11]
y2 = []
l = [i for i in range(len(x))]
for i in y:
#     print(f'{i/sum(y)*100:.4f}%')
    y2.append(i/sum(y)*100)

plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

fmt='%.2f%%'
yticks = mtick.FormatStrFormatter(fmt)  #设置百分比形式的坐标轴

plt.style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(111)
l1 = plt.bar(l,y,alpha=0.3,label='Number of instances')
# for i,(_x,_y) in enumerate(zip(l,y)):
#     plt.text(_x,_y+5,y[i],rotation=90,fontsize=12)  #将数值显示在图形上
# for x1,yy in zip(l,y):
#     plt.text(x1, yy, yy,ha='center', va='center', rotation=90)
ax1.set_ylabel('Instances')
ax1.set_xlabel('Attack Labels')


ax2 = ax1.twinx()# 复制轴
l2 = ax2.plot(l,y2,label = "% of prevelance w.r.t. the total instances")
ax2.yaxis.set_major_formatter(yticks)
ax2.set_ylim([0,100])
ax2.set_ylabel('Rate')




plt.xticks(l,x)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90,fontsize=8)
plt.title("CICIDS2017 Dataset")
fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)# 合并图例！！！！
plt.tight_layout()
plt.savefig("graph.png")
plt.show()

print(1)

