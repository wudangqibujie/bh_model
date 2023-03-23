# # import torch
# # import numpy as np
# # dims = [2, 3, 5, 2, 4]
# # res = np.cumsum(dims)
# # print(res)
# # offsets = np.array((0, *np.cumsum(dims)[:-1]), dtype=np.long).unsqueeze(0)
# # print(offsets)
# import pandas as pd
#
# file_path = "data/byterec_sample.txt"
# data = pd.read_csv(file_path, sep='\t',
#                    names=["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like",
#                           "music_id", "device", "time", "duration_time"])
# y = data["finish"]
# X = data.drop(columns=["finish", "like"])
#
# dense_feats = ["time", "duration_time"]
# sparse_feats = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id",
#                 "device"]
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#
#
# feat_dim_info = []
# for feat in sparse_feats:
#     lbe = LabelEncoder()
#     X[feat] = lbe.fit_transform(X[feat])
#     print(feat, X[feat].min(), X[feat].max())
#     feat_dim_info.append(X[feat].max() + 1)
# nms = MinMaxScaler(feature_range=(-1, 1))
# X[dense_feats] = nms.fit_transform(X[dense_feats])
# feats = sparse_feats + dense_feats
# feat_dim_info.extend([1, 1])
# print(feat_dim_info)
from torch.nn import Transformer
# # df = X
# # df["label"] = y
# # df.to_csv("data/byterec_sample.csv", index=False)
from deepctr_torch.models import DIEN
# log_writer.add_scalar(tag="train/loss")

# ------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names, build_input_features
from deepctr_torch.models import DeepFM


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


data = pd.read_csv("./data/movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]
target = ['rating']

# 1.Label Encoding for sparse features,and process sequence features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
# preprocess the sequence feature
key2index = {}
genres_list = list(map(split, data['genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
# Notice : padding=`post`
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

# 2.count #unique features for each sparse field and generate feature config for sequence feature
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                          for feat in sparse_features]

varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
    key2index) + 1, embedding_dim=4), maxlen=max_len,
                                           combiner='mean')]  # Notice : value 0 is for padding for sequence input feature

linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(build_input_features(linear_feature_columns + dnn_feature_columns))
print(feature_names)
# 3.generate input data for model
model_input = {name: data[name] for name in sparse_features}  #
model_input["genres"] = genres_list

# 4.Define Model,compile and train

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)

model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(model_input, data[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)



# ------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
#
# data description can be found in https://www.biendata.xyz/competition/icmechallenge2019/
data = pd.read_csv('data//byterec_sample.txt', sep='\t',
                   names=["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like",
                          "music_id", "device", "time", "duration_time"])

sparse_features = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
dense_features = ["duration_time"]

target = ['finish', 'like']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 2.count #unique features for each sparse field,and record dense feature field name

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                          for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                          for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(
    linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model

split_boundary = int(data.shape[0] * 0.8)
train, test = data[:split_boundary], data[split_boundary:]
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
#
model = MMOE(dnn_feature_columns, task_types=['binary', 'binary'],
             l2_reg_embedding=1e-5, task_names=target, device=device)
model.compile("adagrad", loss=["binary_crossentropy", "binary_crossentropy"],
              metrics=['binary_crossentropy'], )
#
history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2)
pred_ans = model.predict(test_model_input, 256)
print("")
for i, target_name in enumerate(target):
    print("%s test LogLoss" % target_name, round(log_loss(test[target[i]].values, pred_ans[:, i]), 4))
    print("%s test AUC" % target_name, round(roc_auc_score(test[target[i]].values, pred_ans[:, i]), 4))

# # ------------------------------------------------------------------------------------------------------------------------------
# import sys
# sys.path.insert(0, '..')
# import numpy as np
# import torch
# from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
#                                   get_feature_names)
# from deepctr_torch.models.din import DIN
#
# def get_xy_fd():
#     feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat('gender', 2, embedding_dim=8),
#                        SparseFeat('item', 3 + 1, embedding_dim=8), SparseFeat('item_gender', 2 + 1, embedding_dim=8),
#                        DenseFeat('score', 1)]
#
#     feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), 4, length_name="seq_length"),
#                         VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), 4, length_name="seq_length")]
#     behavior_feature_list = ["item", "item_gender"]
#     uid = np.array([0, 1, 2])
#     ugender = np.array([0, 1, 0])
#     iid = np.array([1, 2, 3])  # 0 is mask value
#     igender = np.array([1, 2, 1])  # 0 is mask value
#     score = np.array([0.1, 0.2, 0.3])
#
#     hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
#     hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
#     behavior_length = np.array([3, 3, 2])
#
#     feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
#                     'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score,
#                     "seq_length": behavior_length}
#     x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
#     y = np.array([1, 0, 1])
#
#     return x, y, feature_columns, behavior_feature_list
#
#
#
# x, y, feature_columns, behavior_feature_list = get_xy_fd()
# device = 'cpu'
# use_cuda = True
# if use_cuda and torch.cuda.is_available():
#     print('cuda ready...')
#     device = 'cuda:0'
#
# model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
# model.compile('adagrad', 'binary_crossentropy',
#               metrics=['binary_crossentropy'])
# history = model.fit(x, y, batch_size=3, epochs=10, verbose=2, validation_split=0.0)