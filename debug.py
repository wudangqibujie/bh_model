# from config.parse_config import MyConfigParser
# import data_loader.data_loaders as data_module
# import model.model as module_module
# from utils import prepare_device
# import torch
#
# config = MyConfigParser.from_config('config/config.json')
# logger = config.get_logger('train')
# train_dataset = config.init_obj('train_data_loader', data_module)
# valid_dataset = config.init_obj('valid_data_loader', data_module)
# model = config.init_obj('arch', module_module)
# logger.info(model)
# device, device_ids = prepare_device(config['n_gpu'])
# model = model.to(device)
# if len(device_ids) > 1:
#     model = torch.nn.DataParallel(model, device_ids=device_ids)
# # from model.LR import LogisticRg
#
# lr = LogisticRg(128)
#
# for i in lr.reg_items:
#     print(type(i))
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
data = pd.read_csv("./data/movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]
target = ['rating']
def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
key2index = {}
genres_list = list(map(split, data['genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
