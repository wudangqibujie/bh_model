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

from features.features import FeatureInfo
featureInfo = FeatureInfo.from_config('features/feature_info.json')
