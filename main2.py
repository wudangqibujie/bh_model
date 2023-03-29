import pandas as pd
from features.features import MyFeatureInfo
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from trainer import Trainer
from config.parse_config import MyConfigParser
from utils import prepare_device
import torch
from data_loader.data_loaders import ScoreDataLoader


SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

data = pd.read_csv("./data/movielens_sample.csv")
features_columns = ["movie_id", "user_id", "gender", "age", "occupation", "zip", "genres_length", "genres"]
target = ['rating']
feature_info = MyFeatureInfo.from_config('./features/feature_info.json')
feature_info.set_feature_idx(features_columns)
print(feature_info["movie_id"])


config = MyConfigParser.from_json('config/config.json')
print(config["valid_data_loader"])
logger = config.get_logger('train')
# data_loader = config.init_obj('train_data_loader', module_data)




if __name__ == '__main__':

    for X, y in data_loader:
        print(X)
        print(y)


