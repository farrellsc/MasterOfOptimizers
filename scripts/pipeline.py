from MasterOfOptimizers.classifier.LinearRegression import LinearRegression
from MasterOfOptimizers.optimizer.MiniBatchGD import MiniBatchGD
from MasterOfOptimizers.dataloader.BaseDataloader import BaseDataloader


def main():
    dataloader = BaseDataloader(
        file_path=""
    )
    optimizer = MiniBatchGD(
        batch_size=1
    )
    model = LinearRegression(
        optimizer=optimizer
    )
    model.train(dataloader.train_set)
    model.analyze()


if __name__ == '__main__':
    main()
