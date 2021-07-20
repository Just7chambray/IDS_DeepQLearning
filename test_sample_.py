import json
import numpy as np
from keras.models import model_from_json
from AD_sample import RLenv
import matplotlib.pyplot as plt


import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import  confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    test_path = 'data/CICIDS2017/Simple/sample/Simple_sample.csv'
    ddos_path ="./data/CICIDS2017/Simple/ddos_data/Dddos.csv"
    portscan_path = "./data/CICIDS2017/Simple/portscan_data/Dportscan.csv"
    attack_path = "./data/CICIDS2017/Simple/attack_data/Dattack.csv"

    model_weights = "./data/CICIDS2017/Simple/ddos_data/model/model.h5"
    model_json = "./data/CICIDS2017/Simple/ddos_data/model/model.json"

    with open(model_json,"r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights(model_weights)
    model.compile('sgd','mse')

    batch_size = 64

    env = RLenv(ddos_path,'test',batch_size)

    true_ones = 0
    true_zeros = 0
    false_ones = 0
    false_zeros = 0
    total_reward = 0
    all_labels = np.array([])
    all_actions=np.array([])
    epochs = int(env.state_shape[0]/env.batch_size)

    for e in range(epochs):
        states, labels = env.get_batch(batch_size=env.batch_size)
        q = model.predict(states)
        actions = np.argmax(q,axis=1)

        all_labels = np.append(all_labels,labels)
        all_actions = np.append(all_actions,actions)

        reward = np.zeros(env.batch_size)
        for indx,a in enumerate(actions):
            if a == labels[indx]:
                reward[indx] = 1
                if a == 0:
                    true_zeros += 1
                else:
                    true_ones += 1
            else:
                if a == 0:
                    false_zeros += 1
                else:
                    false_ones += 1
        total_reward += int(sum(reward))

        print("\rEpoch {}/{} | Ones/Zeros: {}/{}  Tot Rew -- > {}".format(e,epochs,true_ones,true_zeros,total_reward))

    print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward,
            int(epochs*env.batch_size),float(100*total_reward/(epochs*env.batch_size))))


    ind = np.arange(1,5)
    fig, ax = plt.subplots()

    t_o, f_o, t_z ,f_z = plt.bar(ind, (true_ones,false_ones,true_zeros,false_zeros))

    t_o.set_facecolor('g')
    f_o.set_facecolor('r')
    t_z.set_facecolor('g')
    f_z.set_facecolor('r')

    ax.set_xticks(ind)
    ax.set_xticklabels(['True ones', 'False ones', 'True zeros', 'False zeros'])
    # ax.set_yscale('log')

    #ax.set_ylim([0, 100])
    #ax.set_ylabel('Percent usage')
    ax.set_title('Test set')
    plt.tight_layout()
    plt.show()
    plt.savefig('./data/CICIDS2017/Simple/ddos_data/results/test_simple_nat.eps', format='eps', dpi=1000)


    #%% Agregated precision
    print('Performance measures on Test data')
    print('Accuracy =  {}'.format(accuracy_score(all_labels,all_actions)))
    print('F1 =  {}'.format(f1_score(all_labels,all_actions)))
    print('Precision_score =  {}'.format(precision_score(all_labels,all_actions)))
    print(f'recall_score =  {recall_score(all_labels,all_actions):.2f}')

    cnf_matrix = confusion_matrix(all_labels,all_actions)
    np.set_printoptions(precision=2)
    plt.figure()
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Normal','Atack'], normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
    plt.savefig('./data/CICIDS2017/Simple/ddos_data/results/confusion_matrix_simple.png', format='svg', dpi=1000)