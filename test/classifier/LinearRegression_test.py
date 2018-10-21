from MasterOfOptimizers.classifier.LinearRegression import LinearRegression
from MasterOfOptimizers.util.BaseTestCase import BaseTestCase
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader


class TestLinearRegression(BaseTestCase):
    def setUp(self):
        self.dataloader = BaseDataloader(
            file_path=""
        )
        optimizer = MiniBatchGD(
            batch_size=1
        )
        self.model = LinearRegression(
            optimizer=optimizer
        )
        raise NotImplementedError

    def test_training(self):
        self.model.train(self.dataloader.train_set)
        raise NotImplementedError
