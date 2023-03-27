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
import warnings
warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names, build_input_features, DenseFeat
from deepctr_torch.models import DeepFM
from collections import OrderedDict


# data = pd.read_csv("./data/movielens_sample.txt")
# sparse_features = ["movie_id", "user_id",
#                    "gender", "age", "occupation", "zip", ]
# target = ['rating']
# print(data.columns)
# for feat in sparse_features:
#     lbe = LabelEncoder()
#     data[feat] = lbe.fit_transform(data[feat])
# print(data.head())
# def split(x):
#     key_ans = x.split('|')
#     for key in key_ans:
#         if key not in key2index:
#             key2index[key] = len(key2index) + 1
#     return list(map(lambda x: key2index[x], key_ans))
# key2index = {}
# genres_list = list(map(split, data['genres'].values))
# genres_length = np.array(list(map(len, genres_list)))
# max_len = max(genres_length)
# print(key2index)
# data["genres_length"] = data["genres"].apply(lambda x: len(x.split('|')))
# data["genres"] = data["genres"].apply(lambda x: ",".join([str(key2index[i]) for i in x.split('|')]))
# print(data["genres_length"])
# data.to_csv('./data/movielens_sample.csv', index=False)

data = pd.read_csv("./data/movielens_sample.csv")
features_columns = ["movie_id", "user_id", "gender", "age", "occupation", "zip", "genres_length", "genres"]
target = ['rating']
print(data.columns)
from features.features import FeatureInfo
feature_info = FeatureInfo.from_config('./features/feature_info.json')
feature_info.set_feature_idx(features_columns)
print(feature_info.info)





# genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
#
# fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
#                           for feat in sparse_features]
# varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
#     key2index) + 1, embedding_dim=4), maxlen=max_len,
#                                            combiner='mean')]
# linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
# dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
# # feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
# features = build_input_features(linear_feature_columns + dnn_feature_columns)
# print(features)
# feature_names = list(features.keys())
# print(feature_names)
# model_input = {name: data[name] for name in sparse_features}  #
# model_input["genres"] = genres_list
# print(linear_feature_columns)
# print(dnn_feature_columns)
# device = 'cpu'
# use_cuda = True
# if use_cuda and torch.cuda.is_available():
#     print('cuda ready...')
#     device = 'cuda:0'
