import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
start = time.time()


#get mnist data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
data_x = np.array(train.iloc[:, 1:])
data_y = np.array(train.iloc[:, 0])

for svc in ['rbf', 'sigmoid', 'poly']:

    accuracies = []

    print ('----------------------------------------------------')
    ns = [10,20,40]
    for pca_n in ns:
    	#get pac feature
    	pca = PCA(n_components=pca_n, whiten=True)
    	pca.fit(data_x)
    	x_pca = pca.transform(data_x)
    	#make split
    	x_train, x_val, y_train, y_val = train_test_split(x_pca, data_y, test_size=0.2, stratify=data_y)
    	#train process
        start_time = time.time()
        model = svm.SVC(kernel=svc, C=10)
        model.fit(x_train, y_train)
        score = accuracy_score(y_val, model.predict(x_val))
        print ('svc = ', svc, 'pca_n = ', pca_n, "accuracy = ", score)
        print (classification_report(y_val, model.predict(x_val)))
        accuracies.append(score)
        end_time = time.time()
        print ('time', end_time-start_time)
        print ('')

    plt.plot(ns, accuracies, label = 'svc = '+svc)


#plt.title('pca n = '+ str(pca_n))
plt.ylabel('accuracy')
plt.xlabel('pca n')
plt.legend(loc='upper right')
plt.show()