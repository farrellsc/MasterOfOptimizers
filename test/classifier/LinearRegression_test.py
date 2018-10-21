from MasterOfOptimizers.classifier.LinearRegression import LinearRegression
from MasterOfOptimizers.util.BaseTestCase import BaseTestCase
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader


class TestLinearRegression(BaseTestCase):
    def setUp(self):
        self.trainDataloader = BaseDataloader(
            file_path="",
            batch_size=5
        )
        optimizer = MiniBatchGD(
            batch_size=1
        )
        self.model = LinearRegression(
            optimizer=optimizer
        )
        raise NotImplementedError

    def test_training(self):
        self.model.train(self.trainDataloader.data)
        self.model.analyze()
        raise NotImplementedError
