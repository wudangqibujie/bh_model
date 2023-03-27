# file_p = "../data/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"
# log = []
# import numpy as np
# total = []
# for ix, line in enumerate(open(file_p, encoding="utf-8")):
#     if ix % 200000 == 0:
#         print(ix)
#     if ix == 0:
#         continue
#     s = line.split()
#     total.append(s[1: ])
#
#
# embe = np.array(total, dtype=np.float)
# np.save("../data/tencent-ailab-embedding-zh-d100-v0.2.0-s/tenct.npy", embe)
# # emb = np.load("../data/tencent-ailab-embedding-zh-d100-v0.2.0-s/tenct.npy")
# # print(emb)

# from features.features import FeatureInfo
# featureInfo = FeatureInfo.from_config('features/feature_info.json')
import yaml

with open('features/feature.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

from dataclasses import dataclass

@dataclass
class SparseFeat:
    name: str
    vocabulary_size: int
    embedding_dim: int
    use_hash: bool
    dtype: str

    def __post_init__(self):
        if self.embedding_dim == "auto":
            self.embedding_dim = 6 * int(pow(self.vocabulary_size, 0.25))

config = SparseFeat(**config)
print(config)
