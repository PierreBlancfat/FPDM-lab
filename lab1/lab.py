import numpy as np
import re
import os
from sklearn import mixture
from scipy import stats
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from spherecluster import VonMisesFisherMixture


def read_data(letter: str):
    path = './Unistroke/Amerge.txt'
    with open(path, 'r') as f:
        nb_line = sum(1 for _ in f)
    with open(path, 'r') as f:
        data = np.zeros((nb_line, 2))
        i = 0
        for line in f:
            data[i][0] = float(line.strip().split(" ")[0])
            data[i][1] = float(line.strip().split(" ")[1])
            i += 1
        return data


def read_data_nonAmerge(letter: str):
    path = './Unistroke/E05.txt'
    with open(path, 'r') as f:
        nb_line = sum(1 for _ in f)
    with open(path, 'r') as f:
        data = np.zeros((nb_line, 2))
        i = 0
        for line in f:
            data[i][0] = float(line.strip().split("\t")[0])
            data[i][1] = float(line.strip().split("\t")[1])
            i += 1
        return data


def unistroke_to_angular(uni_data):
    nb_lines = len(uni_data)
    ang_data = np.zeros((nb_lines, 2))
    for index in range(nb_lines):
        ang_data[index][0] = np.sqrt(uni_data[index][0] ** 2 + uni_data[index][1] ** 2)
        if uni_data[index][0] != 0:
            ang_data[index][1] = np.arctan(uni_data[index][1] / uni_data[index][0])
        else:
            ang_data[index][1] = np.pi / 2
    return ang_data


def mandatory_questions():
    # 1.2.1
    # ang_data = unistroke_to_angular(read_data(""))
    # plt.hist(ang_data[:, 0], 150, density=True)
    # plt.hist(ang_data[:, 1], 150, density=True)

    # 1.2.5
    mix = VonMisesFisherMixture(2)
    data = unistroke_to_angular(read_data(""))
    mix = mix.fit(data)
    # mix.weights_, mix.cluster_centers_, mix.labels


def main():
    # # Preparatory work
    # # # 2
    # mean1 = [-3, 0]
    # cov1 = [[5, -2], [-2, 1]]
    #
    # mean2 = [3, 0]
    # cov2 = [[5, 2], [2, 2]]
    #
    # d1 = np.random.multivariate_normal(mean1, cov1, 500)
    # d2 = np.random.multivariate_normal(mean2, cov2, 500)
    #
    # mixt = np.concatenate((d1, d2))
    # plt.plot(mixt[:, 0], mixt[:, 1], ".")
    # plt.show()
    #
    # ## 3
    # data = read_data("")
    # plt.plot(data[:, 0], data[:, 1], '.')
    # plt.savefig("Amerge_circle")

    # # Data analysis: Gaussian model
    #
    # # 1
    # gmm = mixture.GaussianMixture(2)
    # gmm.fit(data)
    # print("means : ", gmm.means_)
    # print("\n")
    # print("covs : ", gmm.covariances_)
    #
    # # 2
    # # meshgrid
    # plt.clf()
    # labels = gmm.predict(data)
    # x = np.linspace(-2, 2, 500)
    # X, Y = np.meshgrid(x, x)
    # pos = np.empty(X.shape + (2,))
    # pos[:, :, 0] = X
    # pos[:, :, 1] = Y
    # # pdf
    # rv = stats.multivariate_normal(gmm.means_[0], gmm.covariances_[0])
    # rv2 = stats.multivariate_normal(gmm.means_[1], gmm.covariances_[1])
    # # color with classification
    # bool_label = list(map(bool, labels))
    # inv_bool_label = [not i for i in bool_label]
    # data1 = data[bool_label]
    # data2 = data[inv_bool_label]
    # # plot
    # plt.subplot(1, 2, 1)
    # plt.contourf(X, Y, rv.pdf(pos))
    # plt.plot(data1[:, 0], data1[:, 1], '.', 'r')
    # plt.plot(data2[:, 0], data2[:, 1], '.', 'b')
    #
    # plt.subplot(1, 2, 2)
    # plt.contourf(X, Y, rv2.pdf(pos))
    # plt.plot(data1[:, 0], data1[:, 1], '.', 'r')
    # plt.plot(data2[:, 0], data2[:, 1], '.', 'b')
    # # plt.show()
    #
    # # 3.1
    # plt.clf()
    #
    # plt.hist(data[:, 0], 150, density=True)
    # plt.hist(data[:, 1], 150, density=True)
    # plt.show()
    # mg = stats.norm(gmm.means_[0, 0], gmm.covariances_[0, 0, 0])
    # mg2 = stats.norm(gmm.means_[0, 1], gmm.covariances_[0, 1, 1])
    # mg3 = stats.norm(gmm.means_[1, 0], gmm.covariances_[1, 0, 0])
    # mg4 = stats.norm(gmm.means_[1, 1], gmm.covariances_[1, 1, 1])
    #
    # plt.plot(x, (mg.pdf(x) + mg3.pdf(x)) * 0.5)
    # plt.plot(x, (mg2.pdf(x) + mg4.pdf(x)) * 0.5)
    #
    # # plt.show()
    #
    # # 3.2
    # bar_num = 40
    # plt.clf()
    # plt.subplot(2, 2, 1)
    # plt.hist(data1[:, 0], bar_num, density=True)
    # plt.plot(x, (mg3.pdf(x)))
    #
    # plt.subplot(2, 2, 2)
    # plt.hist(data1[:, 1], bar_num, density=True)
    # plt.plot(x, (mg4.pdf(x)))
    #
    # plt.subplot(2, 2, 3)
    # plt.hist(data2[:, 0], bar_num, density=True)
    # plt.plot(x, (mg.pdf(x)))
    #
    # plt.subplot(2, 2, 4)
    # plt.hist(data2[:, 1], bar_num, density=True)
    # plt.plot(x, (mg2.pdf(x)))
    #
    # plt.show()

    ## Mandatory additionnal questions

    mandatory_questions()


if __name__ == '__main__':
    main()
