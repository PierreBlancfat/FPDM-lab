import numpy as np
import re
import os
from sklearn import mixture
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

def read_data(letter : str):

    path = './Unistroke/Amerge.txt'
    with open(path, 'r') as f:
        nb_line = sum(1 for _ in f)
    with open(path, 'r') as f:
        data = np.zeros((nb_line,2))
        i = 0
        for line in f:
            data[i][0] = float(line.strip().split(" ")[0])
            data[i][1] = float(line.strip().split(" ")[1])
            i+=1
        return data


# Preparatory work 

## 2

mean1 = [-3,0]
cov1 = [[5,-2],[-2,1]]

mean2 = [3,0]
cov2 = [[5,2],[2,2]]

d1 = np.random.multivariate_normal(mean1, cov1, 500)
d2 = np.random.multivariate_normal(mean2, cov2, 500)

mixt = np.concatenate((d1,d2))
plt.plot(mixt[:,0], mixt[:,1],".")
plt.show()


## 3 

data = read_data("A")
print(data)
plt.plot(data[:,0], data[:,1],'.')
# plt.show()

gmm = mixture.GaussianMixture(2,'full')
gmm.fit(data)
print("means : ",gmm.means_)
print("\n")
print("cove : ",gmm.covariances_)
