import json
import numpy as np
import pandas as pd
from keras.models import model_from_json
from multiAD_sample import RLenv
import matplotlib.pyplot as plt


import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


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
    test_path = 'data/CICIDS2017/Multiple/sample/Multiple_sample_final.csv'

    multi_path = './data/CICIDS2017/Multiple/multi_data/Dmulti.csv'

    model_weights = "./data/CICIDS2017/Multiple/multi_data/model/model.h5"
    model_json = "./data/CICIDS2017/Multiple/multi_data/model/model.json"

    with open(model_json,"r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights(model_weights)
    model.compile('sgd','mse')

    batch_size = 64

    env = RLenv(multi_path,'test',batch_size)

    total_reward = 0
    epochs = int(env.state_shape[0]/env.batch_size) # 完成一次完整的训练集训练的迭代次数
    true_labels = np.zeros(len(env.attack_names))
    estimated_labels = np.zeros(len(env.attack_names), dtype=int)
    estimated_correct_labels = np.zeros(len(env.attack_names), dtype=int)
    all_labels = np.array([])
    all_actions = np.array([])

    for e in range(epochs):
        states, labels = env.get_batch(batch_size=env.batch_size)
        q = model.predict(states)
        actions = np.argmax(q, axis=1)

        all_labels = np.append(all_labels, np.argmax(labels.values, axis=1))
        all_actions = np.append(all_actions, actions)

        reward = np.zeros(env.batch_size)
        true_labels += np.sum(labels).values

        for indx,a in enumerate(actions):
            estimated_labels[a] += 1
            if a == np.argmax(labels.iloc[indx].values):
                reward[a] = 1
                estimated_correct_labels[a] += 1

        total_reward += int(sum(reward))

        print("\rEpoch {}/{} | Tot Rew -- > {}".format(e,epochs,total_reward),end="")

    # print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward,
    #         int(epochs*env.batch_size),float(100*total_reward/(epochs*env.batch_size))))

    Accuracy = np.nan_to_num(estimated_correct_labels/true_labels)
    Mismatch = abs(estimated_correct_labels-true_labels)+abs(estimated_labels-estimated_correct_labels)

    print('\r\nTotal reward: {} | Number of samples: {} '.format(total_reward,
            int(epochs * env.batch_size)))

    outputs_df = pd.DataFrame(index=env.attack_names, columns=["Estimated", "Correct", "Total", "Acuracy", "Mismatch"])
    for indx,att in enumerate(env.attack_names):
        outputs_df.iloc[indx].Estimated = estimated_labels[indx]
        outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
        outputs_df.iloc[indx].Total = true_labels[indx]
        outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])
        outputs_df.iloc[indx].Acuracy = Accuracy[indx] * 100

    print(outputs_df)

    ind = np.arange(1,len(env.attack_names)+1)
    fig, ax = plt.subplots()
    width = 0.35
    p1 = plt.bar(ind,estimated_correct_labels,width,color='g')
    p2 = plt.bar(ind,
                 (np.abs(estimated_correct_labels - true_labels) \
                  + np.abs(estimated_labels - estimated_correct_labels)), width,
                 bottom=estimated_correct_labels, color='r')

    ax.set_xticks(ind)
    ax.set_xticklabels(env.attack_names, rotation='vertical')
    # ax.set_yscale('log')

    #ax.set_ylim([0, 100])
    #ax.set_ylabel('Percent usage')
    ax.set_title('Test set scores')
    plt.legend((p1[0],p2[0]),('Correct estimated','Incorrect estimated'))
    plt.tight_layout()
    plt.show()
    plt.savefig('./data/CICIDS2017/Multiple/sample/results/test_multiple.eps', format='eps', dpi=1000)


    #%% Agregated precision
    print('Performance measures on Test data')
    print('Accuracy =  {}'.format(accuracy_score(all_labels,all_actions)))
    print('F1 =  {}'.format(f1_score(all_labels,all_actions,average='weighted')))
    print('Precision_score =  {}'.format(precision_score(all_labels,all_actions,average='weighted')))
    print('recall_score =  {}'.format(recall_score(all_labels,all_actions,average='weighted')))

    cnf_matrix = confusion_matrix(all_labels,all_actions)
    np.set_printoptions(precision=2)
    plt.figure()
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=env.attack_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
    plt.savefig('./data/CICIDS2017/Multiple/sample/results/confusion_matrix_multiple.png', format='svg', dpi=1000)