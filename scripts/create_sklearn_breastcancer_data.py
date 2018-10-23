import sklearn.datasets
from sklearn.model_selection import train_test_split
import pickle


def main():
    dataAll = sklearn.datasets.load_breast_cancer()
    data_name = "breast_cancer_569"
    train_set, test_set, train_label, test_label = train_test_split(dataAll['data'], dataAll['target'], test_size=0.2)
    pickle.dump(
        {
            "train_set": train_set,
            "test_set": test_set,
            "train_label": train_label.reshape([-1, 1]),
            "test_label": test_label.reshape([-1, 1])
        },
        open('../data/' + data_name, 'wb')
    )


if __name__ == "__main__":
    main()
