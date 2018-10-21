import pickle


class BaseDataloader:
    def __init__(self, file_path: str, batch_size):
        self.data, self.label = self.load_file(file_path)
        self.batch_size = batch_size
        self.sample_num = self.data.shape[0]
        assert self.sample_num % self.batch_size == 0, "batch size should be divisible by sample_num"
        self.batch_num = self.sample_num // self.batch_size

    @staticmethod
    def load_file(file_path: str):
        data = pickle.load(open(file_path, 'rb'))
        return data['train_set'], data['train_label']

    def __iter__(self):
        for i in range(self.batch_num):
            yield self.data[i*self.batch_size : (i+1)*self.batch_size, :], \
                  self.label[i*self.batch_size : (i+1)*self.batch_size, :]
