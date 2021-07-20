import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import json
from sklearn.utils import shuffle
import os
import sys
from keras.utils import plot_model



class data_cls:
    def __init__(self,path, train_test,**kwargs):

        self.index = 0
        self.headers = None
        self.loaded = False
        # self.formated_path = "./data/CICIDS2017/Simple/sample/Simple_sample_train.csv"
        # self.test_path = "./data/CICIDS2017/Simple/sample/Simple_sample_test.csv"
        self.formated_path = "./data/CICIDS2017/Simple/ddos_data/Dddos_train.csv"
        self.test_path = "./data/CICIDS2017/Simple/ddos_data/Dddos_test.csv"
        self.train_test = train_test
        self.train_nor_max = []
        self.train_nor_min = []


        formated = False

        if os.path.exists(self.formated_path) and train_test =='train':
            formated = True
        elif os.path.exists(self.test_path) and train_test == 'test':
            formated = True

        if not formated:
            self.df = pd.read_csv(path,sep=',')
            # first to split dataset to train & test
            train_index = np.int32(self.df.shape[0] * 0.7)
            # self.df = shuffle(self.df, random_state=42)
            self.df = shuffle(self.df)
            test_df = self.df.iloc[train_index:self.df.shape[0]]
            train_df = self.df[:train_index]
            if train_test == 'train':
                self.df = train_df
                self.df = shuffle(self.df, random_state=np.random.randint(0, 100))
                if 'Flow Bytes/s' in self.df.columns:
                    del(self.df['Flow Bytes/s'])
                if ' Flow Packets/s' in self.df.columns:
                    del (self.df[' Flow Packets/s'])
                for indx,dtype in self.df.dtypes.iteritems():
                    if dtype == 'float64' or dtype == 'int64':
                        self.train_nor_max.append(self.df[indx].max())
                        self.train_nor_min.append(self.df[indx].min())
                        if self.df[indx].max() == 0 and self.df[indx].min() == 0:
                            self.df[indx] = 0
                        else:
                            self.df[indx] = (self.df[indx] - self.df[indx].min()) / (self.df[indx].max() - self.df[indx].min())
                self.df.to_csv(self.formated_path, sep=',', index=False)
            elif train_test == 'test':
                self.df = test_df
                self.df = shuffle(self.df, random_state=np.random.randint(0, 100))
                i = -1
                if 'Flow Bytes/s' in self.df.columns:
                    del(self.df['Flow Bytes/s'])
                if ' Flow Packets/s' in self.df.columns:
                    del (self.df[' Flow Packets/s'])
                for indx,dtype in self.df.dtypes.iteritems():
                    if dtype == 'float64' or dtype == 'int64':
                        i += 1
                        if self.df[indx].max() == 0 and self.df[indx].min() == 0:
                            self.df[indx] = 0
                        else:
                            self.df[indx] = (self.df[indx] - self.train_nor_min[i]) / (self.train_nor_max[i] - self.train_nor_min[i])
                self.df.to_csv(self.test_path, sep=',', index=False)




    def get_shape(self):
        if self.loaded is False:
            self._load_df()
        self.data_shape = self.df.shape

        return self.data_shape

    def _load_df(self):
        if self.train_test == 'train':
            self.df = pd.read_csv(self.formated_path,sep=',')
        else:
            self.df = pd.read_csv(self.test_path,sep=',')
        self.loaded = True

    def get_batch(self,batch_size=100):
        if self.loaded is False:
            self._load_df()

        indexes = list(range(self.index, self.index+batch_size))
        if max(indexes)>self.data_shape[0]-1:
            dif = max(indexes) - self.data_shape[0]
            indexes[len(indexes)-dif-1:len(indexes)] = list(range(dif+1))
            self.index = batch_size-dif
            batch = self.df.iloc[indexes]
        else:
            batch = self.df.iloc[indexes]
            self.index += batch_size

        labels = batch[' Label'].values

        return batch,labels



class RLenv(data_cls):
    def __init__(self,path,train_test,batch_size=10,**kwargs):
        data_cls.__init__(self,path,train_test,**kwargs)
        self.batch_size = batch_size
        self.state_shape = data_cls.get_shape(self)

    def reset(self):
        self.state_numb = 0
        self.states, self.labels = data_cls.get_batch(self,self.batch_size)
        self.total_reward = 0
        self.steps_in_episode = 0

        return self.states.values

    def _update_state(self):
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)


    def act(self,actions):
        self.reward = np.zeros(self.batch_size)
        for indx,a in enumerate(actions):
            if a == self.labels[indx]:
                self.reward[indx] =1
        self._update_state()

        self.done = False

        return self.states, self.reward, self.done



if __name__=="__main__":
    # CICIDS2017_root_path = 'D:/dqn/data/CICIDS2017/MachineLearningCSV'
    # CICIDS2017_path = 'data/CICIDS2017/Simple/sample/Simple_sample.csv'
    ddos_path = "./data/CICIDS2017/Simple/ddos_data/Dddos.csv"
    portscan_path = "./data/CICIDS2017/Simple/portscan_data/Dportscan.csv"
    attack_path = "./data/CICIDS2017/Simple/attack_data/Dattack.csv"

    valid_actions = [0, 1]
    num_actions =len(valid_actions)
    epsilon = 0.1
    num_episodes = 300
    iterations_episode = 100

    decay_rate = 0.99
    gamma = 0.001

    hidden_size = 128
    batch_size = 64

    env = RLenv(ddos_path,'train',batch_size)

    # Network architecture
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(env.state_shape[1],),
                    batch_size=batch_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=1e-2), "mse")
    print(model.summary())
    plot_model(model, to_file="./data/CICIDS2017/Simple/ddos_data/model/model.png", show_shapes=True, show_layer_names=True, rankdir='TB')
    reward_chain = []
    loss_chain = []


    for epoch in range(num_episodes):
        loss = 0.
        total_reward_by_episode = 0
        states = env.reset()

        done = False
        # benign=0，attack=1
        ones = 0
        zeros = 0

        exploration = 1

        q = np.zeros([batch_size,num_actions])
        for i_iteration in range(iterations_episode):
            # here to use decaying-epsilon-greedy policy
            if exploration>0.001:
                exploration = epsilon * decay_rate**(epoch*i_iteration)
            if np.random.rand()<=exploration:
                actions = np.random.randint(0,num_actions,batch_size)
            else:
                q = model.predict(states)
                actions = np.argmax(q,axis=1)

            next_states, reward, done = env.act(actions)
            if next_states.shape[0] !=batch_size:
                break

            q_prime = model.predict(next_states)
            indx = np.argmax(q_prime,axis=1)
            sx = np.arange(len(indx))

            targets = reward + gamma * q[sx,indx]
            q[[sx,actions]] = targets

            loss +=model.train_on_batch(states,q)

            states = next_states

            ones += int(sum(actions))
            zeros += batch_size - int(sum(actions))
            total_reward_by_episode += int(sum(reward))

        if next_states.shape[0] != batch_size:
            break # finished df
        reward_chain.append(total_reward_by_episode)
        loss_chain.append(loss)

        print("\rEpoch {:03d}/{:03d} | Loss {:4.4f} | Tot reward x episode {:03d}| Ones/Zeros: {}/{} ".format(epoch,
            num_episodes ,loss, total_reward_by_episode,ones,zeros))
    print('over')
    model_weights = "./data/CICIDS2017/Simple/ddos_data/model/model.h5"
    model_json = "./data/CICIDS2017/Simple/ddos_data/model/model.json"
    model.save_weights(model_weights, overwrite=True)
    with open(model_json, "w") as outfile:
        json.dump(model.to_json(), outfile)
    # model.save_weights('./data/CICIDS2017/Simple/sample/model/model.h5', overwrite=True)
    # with open('./data/CICIDS2017/Simple/sample/model/model.json', "w") as outfile:
    #     json.dump(model.to_json(), outfile)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.arange(len(reward_chain)),reward_chain)
    plt.title('Total reward by episode')
    plt.xlabel('n Episode')
    plt.ylabel('Total reward')

    plt.subplot(212)
    plt.plot(np.arange(len(loss_chain)),loss_chain)
    plt.title('Loss by episode')
    plt.xlabel('n Episode')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.show()
    plt.savefig('./data/CICIDS2017/Simple/ddos_data/results/train_simple.eps', format='eps', dpi=1000)