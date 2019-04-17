import numpy
import csv
import urllib
from sklearn import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import *
from pybrain.datasets import *
from pybrain.structure.modules import *



def multiple_days_forward(data, days):
    labels = ((data[days:, 3] - data[days:, 0]) > 0).astype(int)
    data = data[:-days, :]
    return data, labels


    
while True:
    data = list()
    print ("Enter Company/Stock: ")
    print ("1. Nifty")
    print ("2. HCL")
    print ("3. Infy")
    print ("4. ONGC")
    print ("5. Reliance")
    print ("6. Exit")
    case = int(input())




    if case == 1:
        url = 'daily_dataset\\Nifty Historical Daily.csv'
    elif case == 2:
        url = 'daily_dataset\\HCL Historical Daily.csv'
    elif case == 3:
        url = 'daily_dataset\\Infy Historical Daily.csv'
    elif case == 4:
        url = 'daily_dataset\\ONGC Historical Daily.csv'
    elif case == 5:
        url = 'daily_dataset\\Reliance Historical Daily.csv'
    elif case == 6:
        break
    with open(url, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = numpy.array(data)
    data = data[1:, 1:]
    data = data.astype(float)
    labels = ((data[:, 3] - data[:, 0]) > 0).astype(int)
    data, labels = multiple_days_forward(data, 1)

    

    def t_high(t, X):
        return max(X[:-t])




    def t_low(t, X):
        return min(X[:-t])




    def volume_high(t, X):
        return max(X[:-t])



    def volume_low(t, X):
        return min(X[:-t])




    def extract_features(data, indices):
        data = data[:, [0, 1, 2, 3, 5]]
        data2 = data[1:, :]
        features = data[:-1] - data2
        Phigh = t_high(5, data[:, 1])
        Plow = t_low(5, data[:, 2])
        vhigh = volume_high(5, data[:, 4])
        vlow = volume_low(5, data[:, 4])
        Odiff_by_highlow = features[:, 0]/ float(Phigh - Plow)
        Cdiff_by_highlow = features[:, 1]/float(Phigh - Plow)
        mov_avg_by_data = list()
        for i in range(len(features)):
            mov_avg_by_data.append(numpy.mean(data[:i+1, :], axis = 0)/data[i, :])
        mov_avg_by_data = numpy.array(mov_avg_by_data)
        features = numpy.column_stack((features, Odiff_by_highlow, Cdiff_by_highlow, mov_avg_by_data))
        return features[:, indices], data



    features, data = extract_features(data, [0, 1, 2, 3, 4])
    train_features = features[:1000]
    test_features = features[1000:]
    train_labels = labels[:1000]
    test_labels = labels[1000:-1]




    clf = svm.SVC(kernel = 'rbf', C = 1.2, gamma = 0.001)
    clf.fit(train_features, train_labels)


    # In[ ]:

    predicted = clf.predict(test_features)
    Accuracy = accuracy_score(test_labels, predicted)
    Precision = recall_score(test_labels, predicted)
    Recall = precision_score(test_labels, predicted)

    print ("Accuracy: ", Accuracy)
    print ("Precision: ", Precision)
    print ("Recall: ", Recall)




    steps = numpy.arange(0, len(test_labels))
    plt.subplot(211)
    plt.xlim(-1, 100)
    plt.ylim(-1, 2)
    plt.ylabel('Actual Values')
    plt.plot(steps, test_labels, drawstyle = 'steps')
    plt.subplot(212)
    plt.xlim(-1, 100)
    plt.ylim(-1, 2)
    plt.xlabel('Days')
    plt.ylabel('Predicted Values')
    plt.plot(steps, predicted, drawstyle = 'steps')
    plt.show()
