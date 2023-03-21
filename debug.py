# import torch
# import numpy as np
# dims = [2, 3, 5, 2, 4]
# res = np.cumsum(dims)
# print(res)
# offsets = np.array((0, *np.cumsum(dims)[:-1]), dtype=np.long).unsqueeze(0)
# print(offsets)
import pandas as pd

file_path = "data/byterec_sample.txt"
data = pd.read_csv(file_path, sep='\t',
                   names=["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like",
                          "music_id", "device", "time", "duration_time"])
y = data["finish"]
X = data.drop(columns=["finish", "like"])

dense_feats = ["time", "duration_time"]
sparse_feats = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id",
                "device"]
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


feat_dim_info = []
for feat in sparse_feats:
    lbe = LabelEncoder()
    X[feat] = lbe.fit_transform(X[feat])
    print(feat, X[feat].min(), X[feat].max())
    feat_dim_info.append(X[feat].max() + 1)
nms = MinMaxScaler(feature_range=(-1, 1))
X[dense_feats] = nms.fit_transform(X[dense_feats])
feats = sparse_feats + dense_feats
feat_dim_info.extend([1, 1])
print(feat_dim_info)

# df = X
# df["label"] = y
# df.to_csv("data/byterec_sample.csv", index=False)