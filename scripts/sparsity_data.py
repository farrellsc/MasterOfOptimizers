import pickle
import numpy as np
import scipy
import random

def load_file(file_path: str):
    data = pickle.load(open(file_path, 'rb'))
    return data['train_set'], data['test_set'], data['train_label'], data['test_label']

def sparsify_data(data, sparsity_filter):
    for instance in data:
        for i in range(len(instance)):
            if (random.random() > sparsity_filter[i]):
                instance[i] = 0.0

    return data

def main():
    file_path = '../data/fake1'
    train_set, test_set, train_label, test_label = load_file(file_path)
    dim = np.shape(test_set)[1]

    # Sparsity filtering
    sparsity_filter = scipy.random.randn(dim) / 3 + 0.8
    print("sparsity_filter:", sparsity_filter)

    train_set = sparsify_data(train_set, sparsity_filter)
    test_set = sparsify_data(test_set, sparsity_filter)
    #print(test_set)
    pickle.dump(
        {
            "train_set": train_set,
            "test_set": test_set,
            "train_label": train_label,
            "test_label": test_label
        },
        open('../data/fake1_sparse', 'wb')
    )

if __name__ == "__main__":
    main()
