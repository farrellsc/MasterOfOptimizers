from MasterOfOptimizers.classifier.LogisticRegression import LogisticRegression
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader
import matplotlib.pyplot as plt


def main():
    trainDataloader = BaseDataloader(
        file_path="../data/fake1",
        batch_size=5
    )

    trainDataloader = BaseDataloader(
        file_path="/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/data/fake1",
        batch_size=100
    )
    iters = 100

    minibatch_model = LogisticRegression(
        optimizer=MiniBatchGD(lr=0.1, momentum=0),
        num_iter=iters
    )
    minibatch_model.train(trainDataloader)
    minibatch_model.plot(
        trainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/minibatch_testcase_boundary.png")

    adagrad_model = LogisticRegression(
        optimizer=AdaGrad(),
        num_iter=iters
    )
    adagrad_model.train(trainDataloader)
    adagrad_model.plot(
        trainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/adagrad_testcase_boundary.png")

    xs = [i+1 for i in range(iters)]
    plt.plot(xs, minibatch_model.get_loss_history())
    plt.plot(xs, adagrad_model.get_loss_history())
    plt.legend(["miniBatch", "adagrad"])
    plt.savefig("/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/loss/multiple_loss.png")


if __name__ == '__main__':
    main()
