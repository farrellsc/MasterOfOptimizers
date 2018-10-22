from MasterOfOptimizers.classifier.LogisticRegression import LogisticRegression
from MasterOfOptimizers.util.BaseTestCase import BaseTestCase
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader


class TestLinearRegression(BaseTestCase):
    def setUp(self):
        self.trainDataloader = BaseDataloader(
            file_path="../../data/fake1",
            batch_size=5
        )
        self.model = LogisticRegression(
            optimizer=None,
            num_iter=1
        )

    def test_training(self):
        self.model.train(self.trainDataloader)
        self.model.plot(self.trainDataloader, "testcase")
