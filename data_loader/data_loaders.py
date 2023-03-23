from base import BaseDataLoader
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import os


class ScoreDataSet(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.y = self.data["label"].values
        self.X = self.data.drop(columns=["label"]).values

    def __getitem__(self, item):
        return self.X[item, :], self.y[item]

    def __len__(self):
        return self.y.shape[0]


class ScoreDataSetIterable(IterableDataset):
    def __init__(self, data_dir, batch_size=32, drop_last=False):
        self.data_dir = data_dir
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __iter__(self):
        for file_name in os.listdir(self.data_dir):
            if "csv" not in file_name:
                continue
            file_path = os.path.join(self.data_dir, file_name)
            for chunk in pd.read_csv(file_path, chunksize=self.batch_size):
                if self.drop_last and  chunk.shape[0] < self.batch_size:
                    continue
                yield self._map_func(chunk)

    def _map_func(self, df_chunk):
        y = df_chunk["label"].values
        X = df_chunk.drop(columns=["label"]).values
        return X, y


class ScoreDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = ScoreDataSet(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
