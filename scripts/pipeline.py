from MasterOfOptimizers.classifier.LogisticRegression import LogisticRegression
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.optimizer.AdaGrad import AdaGrad
from MasterOfOptimizers.optimizer.AdaDelta import AdaDelta
from MasterOfOptimizers.optimizer.RMSProp import RMSProp
from MasterOfOptimizers.optimizer.ADAM import ADAM
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader
import matplotlib.pyplot as plt


def main():
    data_path = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/data/breast_cancer_569"
    trainDataloader = BaseDataloader(
        file_path=data_path,
        batch_size=10
    )
    stochasticTrainDataloader = BaseDataloader(
        file_path=data_path,
        batch_size=1
    )
    fullTrainDataloader = BaseDataloader(
        file_path=data_path,
        batch_size=-1
    )
    iters = 200

    # --------------------------------------------------------------------------------------------------------
    print("full batch gradient descent")
    fullbatch_model = LogisticRegression(
        optimizer=MiniBatchGD(lr=0.1, momentum=0),
        num_iter=iters
    )
    fullbatch_model.train(fullTrainDataloader)
    fullbatch_model.plot(
        fullTrainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/fullbatch_testcase_boundary.png")

    # --------------------------------------------------------------------------------------------------------
    print("stochastic batch gradient descent")
    stochastic_model = LogisticRegression(
        optimizer=MiniBatchGD(lr=0.1, momentum=0),
        num_iter=iters
    )
    stochastic_model.train(stochasticTrainDataloader)
    stochastic_model.plot(
        stochasticTrainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/stochastic_testcase_boundary.png")

    # --------------------------------------------------------------------------------------------------------
    print("mini batch gradient descent")
    minibatch_model = LogisticRegression(
        optimizer=MiniBatchGD(lr=0.1, momentum=0),
        num_iter=iters
    )
    minibatch_model.train(trainDataloader)
    minibatch_model.plot(
        trainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/minibatch_testcase_boundary.png")

    # --------------------------------------------------------------------------------------------------------
    print("mini batch gradient descent with momentum")
    momentum_model = LogisticRegression(
        optimizer=MiniBatchGD(lr=0.1, momentum=0.5),
        num_iter=iters
    )
    momentum_model.train(trainDataloader)
    momentum_model.plot(
        trainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/momentum_testcase_boundary.png")

    # --------------------------------------------------------------------------------------------------------
    print("adagrad")
    adagrad_model = LogisticRegression(
        optimizer=AdaGrad(),
        num_iter=iters
    )
    adagrad_model.train(trainDataloader)
    adagrad_model.plot(
        trainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/adagrad_testcase_boundary.png")

    # --------------------------------------------------------------------------------------------------------
    print("adadelta")
    adadelta_model = LogisticRegression(
        optimizer=AdaDelta(),
        num_iter=iters
    )
    adadelta_model.train(trainDataloader)
    adadelta_model.plot(
        trainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/adadelta_testcase_boundary.png")

    # --------------------------------------------------------------------------------------------------------
    print("rmsprop")
    rmsprop_model = LogisticRegression(
        optimizer=RMSProp(),
        num_iter=iters
    )
    rmsprop_model.train(trainDataloader)
    rmsprop_model.plot(
        trainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/rmsprop_testcase_boundary.png")

    # --------------------------------------------------------------------------------------------------------
    print("adam")
    adam_model = LogisticRegression(
        optimizer=ADAM(),
        num_iter=iters
    )
    adam_model.train(trainDataloader)
    adam_model.plot(
        trainDataloader,
        "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/adam_testcase_boundary.png")

    # --------------------------------------------------------------------------------------------------------
    xs = [i+1 for i in range(iters)]
    plt.plot(xs, stochastic_model.get_loss_history())
    plt.plot(xs, fullbatch_model.get_loss_history())
    plt.plot(xs, minibatch_model.get_loss_history())
    plt.plot(xs, momentum_model.get_loss_history())
    plt.plot(xs, adagrad_model.get_loss_history())
    plt.plot(xs, adadelta_model.get_loss_history())
    plt.plot(xs, rmsprop_model.get_loss_history())
    plt.plot(xs, adam_model.get_loss_history())
    plt.ylim(ymin=0)
    plt.title("Optimizer Comparison")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend(["stochastic", "fullBatch", "miniBatch", "momentum", "adagrad", "adadelta", "rmsprop", "adam"])
    plt.savefig("/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/MasterOfOptimizers/plots/loss/multiple_loss.png")


if __name__ == '__main__':
    main()
