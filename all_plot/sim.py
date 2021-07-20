from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick

Accuracy_knn = 98.5538/100
Precision_knn = 98.5566/100
Recall_knn = 98.5537/100
F1_score_knn = 98.5630/100

Accuracy_lr = 92.0124/100
Precision_lr = 92.0253/100
Recall_lr = 92.0044/100
F1_score_lr = 92.2155/100

Accuracy_nb = 80.5137/100
Precision_nb = 80.5810/100
Recall_nb = 79.8524/100
F1_score_nb = 85.3106/100

Accuracy_dqn = 0.999970
Precision_dqn = 0.999939
Recall_dqn = 1.0000
F1_score_dqn = 0.999969

Algorithm_x = ['KNN','LR','NB','DQN']
Accuracy_y = [Accuracy_knn, Accuracy_lr, Accuracy_nb, Accuracy_dqn]
F1_score_y = [F1_score_knn, F1_score_lr, F1_score_nb, F1_score_dqn]
Precision_y = [Precision_knn, Precision_lr, Precision_nb, Precision_dqn]
Recall_y = [Recall_knn, Recall_lr, Recall_nb, Recall_dqn]

# sange 
# Algorithm_x = ['KNN','RF','DQN']
# Accuracy_y = [Accuracy_knn, Accuracy_lr, Accuracy_dqn]
# F1_score_y = [F1_score_knn, F1_score_rf, F1_score_dqn]
# Precision_y = [Precision_knn, Precision_rf, Precision_dqn]
# Recall_y = [Recall_knn, Recall_rf, Recall_dqn]

plt.style.use("ggplot")# 1.fivethirtyeight 2.ggplot
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率


x_indexes = np.arange(len(Algorithm_x))
# print(x_indexes)
height = 0.2
# plt.bar(x_indexes, Accuracy_y, label="Accuracy")

plt.bar(x_indexes-1*height, Accuracy_y, height, label="Accuracy")
plt.bar(x_indexes+0*height, Precision_y, height, label="Precision")
plt.bar(x_indexes+1*height, Recall_y,height,  label="Recall")
plt.bar(x_indexes+2*height, F1_score_y, height, label="F1_score")

for i, v in enumerate(Accuracy_y):
    plt.text(x_indexes[i] - 1*height, v, str(f'{v*100:.4f}')+"%",fontsize=8,rotation=90)
for i, v in enumerate(Precision_y):
    plt.text(x_indexes[i] - 0*height, v, str(f'{v*100:.4f}')+"%",fontsize=8,rotation=90)
for i, v in enumerate(Recall_y):
    plt.text(x_indexes[i] + 1*height, v, str(f'{v*100:.4f}')+"%",fontsize=8,rotation=90)
for i, v in enumerate(F1_score_y):
    plt.text(x_indexes[i] + 2*height, v, str(f'{v*100:.4f}')+"%",fontsize=8,rotation=90)

plt.xticks(x_indexes,Algorithm_x)
plt.ylim(0.7,1.00)
# xlocs, xlabs = plt.xticks()
# print(xlocs)
# print(xlabs)
plt.title("Anomaly Detection Results Comparing")
plt.xlabel("Algorithms")
plt.ylabel("Persent")
plt.legend(loc=3, bbox_to_anchor=(1.05,0),borderaxespad = 0.)
plt.tight_layout()
# plt.savefig('compare.png',bbox_inches='tight')
plt.show()