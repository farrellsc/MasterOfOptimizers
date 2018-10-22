from MasterOfOptimizers.util.BaseTestCase import BaseTestCase
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader


class TestDataloader(BaseTestCase):
    def setUp(self):
        self.trainDataloader = BaseDataloader(
            file_path="/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/data/fake1",
            batch_size=1000
        )

    def test_dataIter(self):
        for iter in range(3):
            for ind, (data, label) in enumerate(self.trainDataloader):
                print(ind, data.shape, label.shape)
