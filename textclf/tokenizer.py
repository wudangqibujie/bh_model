import json
import pickle as pkl
import jieba


class Vocab:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, vocab):
        self.vocab = vocab

    def save_as_json(self, save_path):
        with open(save_path, "w") as f:
            f.write(json.dumps(self.vocab))

    def save_as_pkl(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self.vocab, f)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def from_json(cls, file_path):
        vocab = json.load(open(file_path, "rb"))
        return cls(vocab)

    @classmethod
    def from_pkl(cls, file_path):
        vocab = pkl.load(open(file_path, 'rb'))
        return cls(vocab)

    @classmethod
    def from_corpus(cls, data_dir):
        """
        有空再补
        :param data_dir:
        :return:
        """
        pass

    @classmethod
    def from_tencent_pretrained(cls, file_p):
        log = {cls.PAD: 0, cls.UNK: 1}
        for ix, line in enumerate(open(file_p, encoding="utf-8")):
            if ix % 20000 == 0:
                print(ix, line.strip())
            if ix == 0:
                continue
            log[line.split(" ")[0]] = len(log) + 1
        return cls(log)

    @staticmethod
    def get_stop_words(file_p):
        with open(file_p, encoding="utf-8") as f:
            return [i.strip() for i in f.readlines()]


class Tokenizer:
    tokenized_word_cnt = 0
    miss_cnt = 0

    def __init__(self, vocab, max_length, stopwords_file=None):
        self.vocab = vocab
        self.stop_words = self.vocab.get_stop_words(stopwords_file) if stopwords_file else []
        self.max_length = max_length

    def tokenize(self, sentence):
        self.input_ids = []
        self.input_text = []
        lines = list(jieba.lcut(sentence))
        lines = [word for word in lines if word not in self.stop_words][: self.max_length]
        for word in lines:
            if word not in self.vocab.vocab:
                self.input_text.append(self.vocab.UNK)
                self.input_ids.append(self.vocab.vocab[self.vocab.UNK])
                self.miss_cnt += 1
            else:
                self.input_text.append(word)
                self.input_ids.append(self.vocab.vocab[word])
                self.tokenized_word_cnt += 1
        if len(self.input_ids) < self.max_length:
            self.input_ids.extend(
                [self.vocab.vocab[self.vocab.PAD]
                    for _ in range(self.max_length - len(self.input_ids))]
            )
            self.input_text.extend(
                [self.vocab.PAD for _ in range(self.max_length - len(self.input_text))]
            )
            assert len(self.input_ids) == self.max_length
            assert len(self.input_text) == self.max_length

        return self.input_ids
