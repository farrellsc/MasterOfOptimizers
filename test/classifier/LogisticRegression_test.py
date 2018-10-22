from MasterOfOptimizers.classifier.LogisticRegression import LogisticRegression
from MasterOfOptimizers.util.BaseTestCase import BaseTestCase
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader


class TestLinearRegression(BaseTestCase):
    def setUp(self):
        self.trainDataloader = BaseDataloader(
            file_path="/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/data/fake1",
            batch_size=5
        )
        self.model = LogisticRegression(
            optimizer=MiniBatchGD(lr=0.1, momentum=0),
            num_iter=1000
        )

    def test_training(self):
        self.model.train(self.trainDataloader)
        self.model.plot(
            self.trainDataloader,
            "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/testcase_boundary.png")
