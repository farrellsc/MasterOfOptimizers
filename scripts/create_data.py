import numpy as np
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
    data_name = "fake1"
    data1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 4]], [train_sample_num])
<<<<<<< HEAD
    data2 = np.random.multivariate_normal([6, 7], [[2, 2], [1, 3]], [train_sample_num])
    data3 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 4]], [test_sample_num])
    data4 = np.random.multivariate_normal([6, 7], [[2, 2], [1, 3]], [test_sample_num])
=======
    data2 = np.random.multivariate_normal([7, 8], [[2, 2], [1, 3]], [train_sample_num])
    data3 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 4]], [test_sample_num])
    data4 = np.random.multivariate_normal([7, 8], [[2, 2], [1, 3]], [test_sample_num])
>>>>>>> 0e2ad632450092a566aaf6d6ad095f182d9faaac
    pickle.dump(
        {
            "train_set": np.vstack([data1, data2]),
            "test_set": np.vstack([data3, data4]),
            "train_label": np.vstack([np.zeros([train_sample_num, 1]), np.ones([train_sample_num, 1])]),
            "test_label": np.vstack([np.zeros([test_sample_num, 1]), np.ones([test_sample_num, 1])])
        },
        open('../data/' + data_name, 'wb')
    )
    save_plot(data1, data2, data3, data4, data_name)


if __name__ == "__main__":
    main()
