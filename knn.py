import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

import time

#get mnist data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
data_x = np.array(train.iloc[:, 1:])
data_y = np.array(train.iloc[:, 0])


for pca_n in [10,20,40]:
    #get pac feature
    pca = PCA(n_components=pca_n)
    pca.fit(data_x)
    x_pca = pca.transform(data_x)
    #make split
    x_train, x_val, y_train, y_val = train_test_split(x_pca, data_y, test_size=0.2, stratify=data_y)
    accuracies = []

    #train process
    print('----------------------------------------------------')
    for k in range(1, 26, 2):
        start_time = time.time()
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)

        score = model.score(x_val, y_val)
        print ("k = ", k, "accuracy = ", score)
        print (classification_report(y_val, model.predict(x_val)))
        accuracies.append(score)
        end_time = time.time()
        print ('time', end_time-start_time)
        print ('')

    plt.plot(accuracies, label = 'n = '+str(pca_n))


#plt.title('pca n = '+ str(pca_n))
plt.ylabel('accuracy')
plt.xlabel('k value')
plt.legend(loc='upper right')
plt.show()

