# import os
# from pathlib import Path
# from gensim.models import KeyedVectors
#
# print(os.listdir('../data/tencent-ailab-embedding-zh-d100-v0.2.0-s'))
#
# data_dir = '../data/tencent-ailab-embedding-zh-d100-v0.2.0-s'
# embedd_file = 'tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
#
# p = Path(data_dir)
# p = p / embedd_file
# print(p)
#
# wv_from_text = KeyedVectors.load_word2vec_format("new.txt", binary=False)
# print(wv_from_text.get_vector("的"))
#
# # f = open(p, encoding="utf-8")
# #
# # new_size = 200
# #
# # new_lines = []
# # stop = 1
# # for i in f:
# #     new_lines.append(i)
# #     stop += 1
# #     if stop > 21:
# #         break
# # f_new = open("new.txt", "w", encoding="utf-8")
# # f_new.writelines(new_lines)

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('../result_tensorboard')
#
# writer.add_scalar('train/loss', 1212., 1)
# tensorboard --logdir="D:\ML_project\result_tensorboard" --host=127.0.0.1


import torch
from textclf.tokenizer import Vocab, Tokenizer
from data_loader.data_loaders import MultiFilesDataSetInterable, MultiFilesInMemDataSetInterable

vocab = Vocab.from_json('../data/tencet_vocab.json')
tokenizer = Tokenizer(vocab, 10, "textclf/stopwords")
preporcess_line = lambda line: line.strip().split('\t')[0]
map_line = lambda line: tokenizer.tokenize((preporcess_line(line)))


def map_func(line):
    sentence, label = line.strip().split('\t')
    ids = tokenizer.tokenize(sentence)
    label = int(label)
    return ids, label


dataloader = MultiFilesInMemDataSetInterable(
    data_dir="../data/THU",
    glob_rule="train.txt",
    batch_size=32,
    drop_lst=False,
    files_shuffle=True,
    map_line=map_func
)

# X, y = next(dataloader)
# X = torch.Tensor(X)
# y = torch.Tensor(y)

# for inputids, labels in dataloader:
#     X = torch.Tensor(inputids)
#     y = torch.Tensor(labels)
#     print(X.shape, y.shape)
