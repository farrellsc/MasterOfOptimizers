from MasterOfOptimizers.classifier.LogisticRegression import LogisticRegression
from MasterOfOptimizers.util.BaseTestCase import BaseTestCase
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader
import matplotlib.pyplot as plt


class TestOptimizers(BaseTestCase):
    def setUp(self):
        self.trainDataloader = BaseDataloader(
            file_path="/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/data/fake1",
            batch_size=1
        )
        self.iters = 100

    def test_multiple_optimizers(self):
        self.minibatch_model = LogisticRegression(
            optimizer=MiniBatchGD(lr=0.1, momentum=0),
            num_iter=self.iters
        )
        self.minibatch_model.train(self.trainDataloader)
        self.minibatch_model.plot(
            self.trainDataloader,
            "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/minibatch_testcase_boundary.png")

        self.adagrad_model = LogisticRegression(
            optimizer=AdaGrad(),
            num_iter=self.iters
        )
        self.adagrad_model.train(self.trainDataloader)
        self.adagrad_model.plot(
            self.trainDataloader,
            "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/adagrad_testcase_boundary.png")

        xs = [i+1 for i in range(self.iters)]
        plt.plot(xs, self.minibatch_model.get_loss_history())
        plt.plot(xs, self.adagrad_model.get_loss_history())
        plt.legend(["miniBatch", "adagrad"])
        plt.savefig("/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/loss/multiple_loss.png")
