from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

Accuracy_knn = 0.9758970500479983
Precision_knn = 0.918148017200979
Recall_knn = 0.897208613113486
F1_score_knn = 0.9037856767436456

Accuracy_lr = 96.6649/100
Precision_lr = 94.3022/100
Recall_lr = 93.6116/100
F1_score_lr= 93.7660/100

Accuracy_nb = 94.5680/100
Precision_nb= 91.5974/100
Recall_nb = 92.9694/100
F1_score_nb = 91.5511/100

Accuracy_dqn = 98.3557/100
Precision_dqn = 98.3628/100
Recall_dqn= 98.3557/100
F1_score_dqn = 98.3429/100

Algorithm_x = ['KNN','LR','NB','DQN']
Accuracy_y = [Accuracy_knn, Accuracy_lr, Accuracy_nb, Accuracy_dqn]
Precision_y = [Precision_knn, Precision_lr, Precision_nb, Precision_dqn]
Recall_y = [Recall_knn, Recall_lr, Recall_nb, Recall_dqn]
F1_score_y = [F1_score_knn, F1_score_lr, F1_score_nb, F1_score_dqn]

# 三个
# Algorithm_x = ['KNN','RF','DQN']
# Accuracy_y = [Accuracy_knn, Accuracy_lr, Accuracy_dqn]
# F1_score_y = [F1_score_knn, F1_score_rf, F1_score_dqn]
# Precision_y = [Precision_knn, Precision_rf, Precision_dqn]
# Recall_y = [Recall_knn, Recall_rf, Recall_dqn]


plt.style.use("ggplot")# 1.fivethirtyeight 2.ggplot
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
x_indexes = np.arange(len(Algorithm_x))
x_indexes = x_indexes/2
# print(x_indexes)
height = 0.1
# plt.bar(x_indexes, Accuracy_y, label="Accuracy")

plt.bar(x_indexes-1*height, Accuracy_y, height, label="Accuracy")
plt.bar(x_indexes+0*height, Precision_y, height, label="Precision")
plt.bar(x_indexes+1*height, Recall_y,height,  label="Recall")
plt.bar(x_indexes+2*height, F1_score_y, height, label="F1_score")

plt.ylim(0.85,1.0)
for i, v in enumerate(Accuracy_y):
    plt.text(x_indexes[i] - 1*height, v + 0.0, str(f'{v*100:.4f}')+"%",fontsize=8,rotation=90)
for i, v in enumerate(Precision_y):
    plt.text(x_indexes[i] + 0*height, v + 0.0, str(f'{v*100:.4f}')+"%",fontsize=8,rotation=90)
for i, v in enumerate(Recall_y):
    plt.text(x_indexes[i] + 1*height, v + 0.00, str(f'{v*100:.4f}')+"%",fontsize=8,rotation=90)
for i, v in enumerate(F1_score_y):
    plt.text(x_indexes[i] + 2*height, v - 0.0, str(f'{v*100:.4f}')+"%",fontsize=8,rotation=90)

'''3
plt.ylim(0.88,1.02)
for i, v in enumerate(Accuracy_y):
    plt.text(x_indexes[i] - 1.5*height, v + 0.002, str(f'{v*100:.2f}')+"%",fontsize=12,verticalalignment="top",rotation=20)
for i, v in enumerate(F1_score_y):
    plt.text(x_indexes[i] - 0.5*height, v - 0.002, str(f'{v*100:.2f}')+"%",fontsize=12,verticalalignment="center",rotation=20)
for i, v in enumerate(Precision_y):
    plt.text(x_indexes[i] + 0.5*height, v - 0.02, str(f'{v*100:.2f}')+"%",fontsize=12,verticalalignment="center",rotation=20)
for i, v in enumerate(Recall_y):
    plt.text(x_indexes[i] + 1.5*height, v + 0.0, str(f'{v*100:.2f}')+"%",fontsize=12,verticalalignment="top",rotation=20)

'''


plt.xticks(x_indexes,Algorithm_x)

# xlocs, xlabs = plt.xticks()
# print(xlocs)
# print(xlabs)
plt.title("Multiple Anomaly Detection Results Comparing")
plt.xlabel("Algorithms")
plt.ylabel("Persent")
plt.legend(loc=3, bbox_to_anchor=(1.05,0),borderaxespad = 0.)
plt.tight_layout()
# plt.savefig('compare.png',bbox_inches='tight')
plt.show()