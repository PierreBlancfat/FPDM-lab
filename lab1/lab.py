import numpy as np
import re
import os
from sklearn import mixture
from scipy import stats
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from spherecluster import VonMisesFisherMixture



def read_data(letter : str):

    path = './Unistroke/'+letter+'merge.txt'
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

def read_and_merge(letter : str):
    files = os.listdir('./Unistroke/')
    regex = re.compile(r'^'+letter+'[0-9]')
    files = list(filter(regex.search, files))
    data = list()
    for file in files:
        with open("./Unistroke/"+file) as content: 
            for line in content:
                data.append((float(line.strip().split("\t")[0]),float(line.strip().split("\t")[1])))

    return np.array(data)


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
    #
    # ## 2
    #
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
    #
    # plt.plot(mixt[:, 0], mixt[:, 1], ".")
    # plt.title('Merging of all A files')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    #
    # ## 3
    # # data = read_and_merge("A")
    # data = read_data("A")
    # plt.figure()
    # plt.plot(data[:, 0], data[:, 1], '.')
    # plt.title('Merging of all A files after angular transformation')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    #
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
    # fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    #
    # axs[0].contourf(X, Y, rv.pdf(pos))
    # axs[0].plot(data1[:, 0], data1[:, 1], '.', color='g')
    # axs[0].plot(data2[:, 0], data2[:, 1], '.', color='b')
    # axs[0].set_title('fitted distribution of blue data points')
    # axs[0].set_xlabel('x')
    # axs[0].set_ylabel('y')
    #
    # axs[1].contourf(X, Y, rv2.pdf(pos))
    # axs[1].plot(data1[:, 0], data1[:, 1], '.', color='g')
    # axs[1].plot(data2[:, 0], data2[:, 1], '.', color='b')
    # axs[1].set_title('fitted distribution of green data points')
    # axs[1].set_xlabel('x')
    # axs[1].set_ylabel('y')
    #
    # fig.suptitle('classification of data points and \n gaussian pdf estimated by the EM algorithm', fontsize=15)
    # fig.show()
    #
    # # 3.1
    # fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    #
    # axs2[0].hist(data[:, 0], 150, density=True)
    # axs2[1].hist(data[:, 1], 150, density=True)
    #
    # mg = stats.norm(gmm.means_[0, 0], gmm.covariances_[0, 0, 0])
    # mg2 = stats.norm(gmm.means_[0, 1], gmm.covariances_[0, 1, 1])
    # mg3 = stats.norm(gmm.means_[1, 0], gmm.covariances_[1, 0, 0])
    # mg4 = stats.norm(gmm.means_[1, 1], gmm.covariances_[1, 1, 1])
    #
    # axs2[0].plot(x, (mg.pdf(x) + mg3.pdf(x)) * 0.5)
    # axs2[1].plot(x, (mg2.pdf(x) + mg4.pdf(x)) * 0.5)
    # axs2[0].set_title('marginal along x')
    # axs2[1].set_title('marginal along y')
    #
    # axs2[0].set_xlabel('x')
    # axs2[0].set_ylabel('density')
    # axs2[1].set_xlabel('y')
    # axs2[1].set_ylabel('density')
    #
    # fig2.suptitle('marginal histograms and marginal \n estimated gaussian mixtures', fontsize=15)
    # fig2.show()
    #
    # # 3.2
    #
    # fig3, axs3 = plt.subplots(2, 2, figsize=(8, 4), constrained_layout=True)
    # bar_num = 40
    # axs3[0][0].hist(data1[:, 0], bar_num, density=True, color='g')
    # axs3[0][0].plot(x, (mg3.pdf(x)), color='r')
    # axs3[0][0].set_title('marginal along x, green cluster')
    #
    # axs3[0][1].hist(data1[:, 1], bar_num, density=True, color='g')
    # axs3[0][1].plot(x, (mg4.pdf(x)), color='r')
    # axs3[0][1].set_title('marginal along y, green cluster')
    #
    # axs3[1][0].hist(data2[:, 0], bar_num, density=True, color='b')
    # axs3[1][0].plot(x, (mg.pdf(x)), color='r')
    # axs3[1][0].set_title('marginal along x, blue cluster')
    #
    # axs3[1][1].hist(data2[:, 1], bar_num, density=True, color='b')
    # axs3[1][1].plot(x, (mg2.pdf(x)), color='r')
    # axs3[1][1].set_title('marginal along y, blue cluster')
    #
    # axs3[0][0].set_xlabel('x')
    # axs3[0][0].set_ylabel('density')
    # axs3[1][0].set_xlabel('x')
    # axs3[1][0].set_ylabel('density')
    # axs3[0][1].set_xlabel('y')
    # axs3[0][1].set_ylabel('density')
    # axs3[1][1].set_xlabel('y')
    # axs3[1][1].set_ylabel('density')
    #
    # fig3.suptitle('marginal estimated pdf and histogram for each cluster ', fontsize=15)
    # fig3.show()

    mandatory_questions()


if __name__ == '__main__':
    main()

