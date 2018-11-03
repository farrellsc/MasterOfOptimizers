import numpy as np
from scipy import sparse, random, linalg
import matplotlib.pyplot as plt
import pickle


def save_plot(data1, data2, data3, data4, data_name):
    plt.scatter(data1[:, 0], data1[:, 1], color='r')
    plt.scatter(data2[:, 0], data2[:, 1], color='b')
    plt.savefig("../data/" + data_name + "_train.png")
    plt.close()
    plt.scatter(data3[:, 0], data3[:, 1], color='r')
    plt.scatter(data4[:, 0], data4[:, 1], color='b')
    plt.savefig("../data/" + data_name + "_test.png")
    plt.close()


def main():
    train_sample_num = 1000
    test_sample_num = 200
    label_ratio = 1.32
    data_name = "fake1"
    dim = 20
    mean_pos = random.rand(dim) * 10 - 5
    mean_neg = random.rand(dim) * 10 - 5
    temp = random.rand(dim, dim) * 2
    var_pos = np.dot(temp, temp.transpose())
    temp = random.rand(dim, dim) * 2
    var_neg = np.dot(temp, temp.transpose())
    
    var_noise = random.rand(dim, dim) * 1.5
    var_noise = np.dot(var_noise, var_noise.transpose())
    
    data1 = np.random.multivariate_normal(mean_neg, var_neg, [int(train_sample_num * label_ratio)]) + np.random.multivariate_normal(np.zeros(dim), var_noise, [int(train_sample_num * label_ratio)]) 
    data2 = np.random.multivariate_normal(mean_pos, var_pos, [train_sample_num]) + np.random.multivariate_normal(np.zeros(dim), var_noise, [train_sample_num]) 
    data3 = np.random.multivariate_normal(mean_neg, var_neg, [int(test_sample_num * label_ratio)]) + np.random.multivariate_normal(np.zeros(dim), var_noise, [int(test_sample_num * label_ratio)]) 
    data4 = np.random.multivariate_normal(mean_pos, var_pos, [test_sample_num]) + np.random.multivariate_normal(np.zeros(dim), var_noise, [test_sample_num]) 
    #print(data3, data4)
    pickle.dump(
        {
            "train_set": np.vstack([data1, data2]),
            "test_set": np.vstack([data3, data4]),
            "train_label": np.vstack([np.zeros([int(train_sample_num * label_ratio), 1]), np.ones([train_sample_num, 1])]),
            "test_label": np.vstack([np.zeros([int(test_sample_num * label_ratio), 1]), np.ones([test_sample_num, 1])])
        },
        open('../data/' + data_name, 'wb')
    )
    save_plot(data1, data2, data3, data4, data_name)


if __name__ == "__main__":
    main()
