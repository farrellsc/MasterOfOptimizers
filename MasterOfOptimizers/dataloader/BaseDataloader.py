import pickle


class BaseDataloader:
    def __init__(self, file_path: str, batch_size):
        """
        :param file_path:
        :param batch_size: -1 for full
        """
        self.data, self.label = self.load_file(file_path)
        self.batch_size = batch_size if batch_size != -1 else self.data.shape[0]
        self.sample_num = self.data.shape[0] // self.batch_size * self.batch_size
        self.sample_dim = self.data.shape[1]
        self.data = self.data[:self.sample_num, :]
        self.label = self.label[:self.sample_num, :]
        self.batch_num = self.sample_num / self.batch_size
        assert int(self.batch_num) == self.batch_num
        self.batch_num = int(self.batch_num)

    @staticmethod
    def load_file(file_path: str):
        data = pickle.load(open(file_path, 'rb'))
        return data['train_set'], data['train_label']

    def __iter__(self):
        for i in range(0, self.batch_num):
            yield self.data[i*self.batch_size : (i+1)*self.batch_size, :], \
                  self.label[i*self.batch_size : (i+1)*self.batch_size, :]
            if i == self.batch_num-1:
                i = 0
