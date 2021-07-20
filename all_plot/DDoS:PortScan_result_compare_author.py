from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

Accuracy_ddos = 0.9553
Accuracy_portscan = 0.9232
Precision_ddos = 0.9894
Precision_portscan = 0.9356
Recall_ddos = 0.9651
Recall_portscan = 0.9654

Accuracy_ddos_me = 0.9806
Accuracy_portscan_me = 0.9999

Precision_ddos_me = 1.0000
Precision_portscan_me = 1.0000

Recall_ddos_me = 0.9600
Recall_portscan_me = 1.0000

'''
DDoS','PortScan 一起
'''
# Algorithm_x = ['DDoS','PortScan']
#
# Accuracy_y = [Accuracy_ddos, Accuracy_portscan]
# Precision_y = [Precision_ddos, Precision_portscan]
# Recall_y = [Recall_ddos, Recall_portscan]
#
# Accuracy_me_y = [Accuracy_ddos_me, Accuracy_portscan_me]
# Precision_me_y = [Precision_ddos_me, Precision_portscan_me]
# Recall_me_y = [Recall_ddos_me, Recall_portscan_me]
# # plt.xkcd()
# # plt.style.use("fivethirtyeight")
# plt.style.use("ggplot")# 1.fivethirtyeight 2.ggplot
#
# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
#
# x_indexes = np.arange(len(Algorithm_x))
# height = 0.1
# plt.bar(x_indexes-3*height, Accuracy_y, height, label="Accuracy in Paper[17]")
# plt.bar(x_indexes-2*height, Accuracy_me_y, height, label="Accuracy in This Paper")
# plt.bar(x_indexes-0*height, Precision_y, height, label="Precision in Paper[17]")
# plt.bar(x_indexes+1*height, Precision_me_y, height, label="Precision in This Paper")
# plt.bar(x_indexes+3*height, Recall_y, height, label="Recall in Paper[17]")
# plt.bar(x_indexes+4*height, Recall_me_y, height, label="Recall in This Paper")
#
#
# plt.xticks(x_indexes,Algorithm_x)
# plt.ylim(0.92,1.0)
# for i, v in enumerate(Accuracy_y):
#     plt.text(x_indexes[i] - 4*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
# for i, v in enumerate(Accuracy_me_y):
#     plt.text(x_indexes[i] - 3*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
# for i, v in enumerate(Precision_y):
#     plt.text(x_indexes[i] - 1*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
# for i, v in enumerate(Precision_me_y):
#     plt.text(x_indexes[i] + 0*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
# for i, v in enumerate(Recall_y):
#     plt.text(x_indexes[i] + 2*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
# for i, v in enumerate(Recall_me_y):
#     plt.text(x_indexes[i] + 3*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
#
# # xlocs, xlabs = plt.xticks()
# # print(xlocs)
# # print(xlabs)
# plt.title("DDoS/PortScan Detection Results Comparing")
# plt.xlabel("Attack")
# plt.ylabel("Persent")
# # plt.legend()
# plt.legend(loc=3, bbox_to_anchor=(1.05,0),borderaxespad = 0.)
# plt.tight_layout()
# # plt.savefig('compare.png',bbox_inches='tight')
# plt.show()
# print(1)



'''
单独DDOS
'''
# Algorithm_x = ['Accuracy','Precision','Recall']
# paper16 = [Accuracy_ddos,Precision_ddos,Recall_ddos]
# thispaper = [Accuracy_ddos_me,Precision_ddos_me,Recall_ddos_me]
#
#
# plt.style.use("ggplot")# 1.fivethirtyeight 2.ggplot
#
# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
#
# x_indexes = np.arange(len(Algorithm_x))
# x_indexes = x_indexes/2.5
# height = 0.1
# plt.bar(x_indexes-0.5*height, paper16, height, label="Thesis[17]")
# plt.bar(x_indexes+0.5*height, thispaper, height, label="This Thesis")
#
#
# plt.xticks(x_indexes,Algorithm_x)
# plt.ylim(0.92,1.0)
# for i, v in enumerate(paper16):
#     plt.text(x_indexes[i] - 1*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
# for i, v in enumerate(thispaper):
#     plt.text(x_indexes[i] + 0*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
#
#
# plt.title("DDoS Detection Results Comparing")
# plt.xlabel("Evaluation Metrics")
# plt.ylabel("Persent")
#
# plt.legend(loc=3, bbox_to_anchor=(1.05,0),borderaxespad = 0.)
# plt.tight_layout()
#
# plt.show()
# print(1)

'''
单独portscan
'''
Algorithm_x = ['Accuracy','Precision','Recall']
paper16 = [Accuracy_portscan,Precision_portscan,Recall_portscan]
thispaper = [Accuracy_portscan_me,Precision_portscan_me,Recall_portscan_me]


plt.style.use("ggplot")# 1.fivethirtyeight 2.ggplot

plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

x_indexes = np.arange(len(Algorithm_x))
x_indexes = x_indexes/2.5
height = 0.1
plt.bar(x_indexes-0.5*height, paper16, height, label="Thesis[17]")
plt.bar(x_indexes+0.5*height, thispaper, height, label="This Thesis")


plt.xticks(x_indexes,Algorithm_x)
plt.ylim(0.92,1.0)
for i, v in enumerate(paper16):
    plt.text(x_indexes[i] - 1*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)
for i, v in enumerate(thispaper):
    plt.text(x_indexes[i] + 0*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=8,rotation=20)


plt.title("PortScan Detection Results Comparing")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Persent")

plt.legend(loc=3, bbox_to_anchor=(1.05,0),borderaxespad = 0.)
plt.tight_layout()

plt.show()
print(1)






