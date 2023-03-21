from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd


class NCFDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ScoreDataSet(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.y = self.data["label"].values
        self.X = self.data.drop(columns=["label"]).values

    def __getitem__(self, item):
        return self.X[item, :], self.y[item]

    def __len__(self):
        return self.y.shape[0]


class ScoreDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = ScoreDataSet(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

