from MasterOfOptimizers.classifier.LinearRegression import LinearRegression
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader


def main():
    trainDataloader = BaseDataloader(
        file_path="../data/fake1",
        batch_size=5
    )
    optimizer = MiniBatchGD(
        batch_size=1
    )
    model = LinearRegression(
        optimizer=optimizer
    )
    model.train(trainDataloader)
    model.analyze()


if __name__ == '__main__':
    main()
