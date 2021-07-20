# IDS_DeepQNetwork

## 简介
本项目为本人的毕设课题《基于深度强化学习的入侵检测系统研究与实现》，本课题对基于深度强化学习的入侵检测系统最新研究工作[<sup>[1]</sup>](#refer-anchor-1)进行复现，实现一个基于深度Q网络(Deep Q Network, DQN)的入侵检测系统。

## 主要工作
- 实现基于DQN算法的单分类流量检测
  - 对DDoS流量和良性流量进行单分类检测
  - 对PortScan流量和良性流量进行单分类检测
  - 对攻击流量(包含多种攻击类型)和良性流量进行单分类检测
- 实现基于DQN算法的多分类攻击流量检测


## 数据集
本课题采用的是加拿大网络安全研究所提出的用于评估入侵检测系统的数据集CICIDS2017[<sup>[2]</sup>](#refer-anchor-2)。
由于数据集较大，实验时只采取了部分数据，数据预处理后压缩存放在oss上，可通过该[链接🔗](https://ids-dqn-dataset.oss-cn-beijing.aliyuncs.com/ids-dqn-dataset.zip?versionId=CAEQFBiBgMC6gOCO1hciIGUwOTQ0ZjU0ZjkwMTQ2NzBiYjk4MmU5NWFjYjEwNjFl)进行访问下载。

## 实验环境
### python版本
3.6.4
### 所需第三方库及对应版本
- tensorflow == 1.15.4
- pandas == 0.23.0
- numpy == 1.16.0
- keras == 2.2.4
- scikit-learn == 0.19.1
- matplotlib == 2.2.2

## 参考

<div id="refer-anchor-1"></div>

- [1] [Janagam, Anirudh, and Saddam Hossen. "Analysis of network intrusion detection system with machine learning algorithms (deep reinforcement learning algorithm)." (2018).](http://www.diva-portal.org/smash/get/diva2:1255686/FULLTEXT02.pdf)

<div id="refer-anchor-2"></div>

- [2] [Sharafaldin, Iman, Arash Habibi Lashkari, and Ali A. Ghorbani. "Toward generating a new intrusion detection dataset and intrusion traffic characterization." ICISSp. 2018.](https://www.scitepress.org/Papers/2018/66398/66398.pdf)
