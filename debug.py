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

