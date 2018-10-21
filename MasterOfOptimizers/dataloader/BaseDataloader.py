class BaseDataloader:
    def __init__(self, file_path: str):
        self.train_set, self.test_set = self.load_file(file_path)
        raise NotImplementedError

    def load_file(self, file_path: str):
        raise NotImplementedError
