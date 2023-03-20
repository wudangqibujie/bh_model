import os
import jieba
import tensorflow as tf
import json
import collections
import random
from sklearn.feature_extraction.text import TfidfVectorizer

base_dir = "../../data/cnews"


class CnewsDataset:
    def __init__(self):
        random.seed(100)
        self.stop_words = self._read_stopwords()
        self.label_map = dict()
        self.vectorize = self.create_vectorize()
        self.voceb_size = len(self.vectorize.get_feature_names())

    def _read_stopwords(self):
        with open(os.path.join(base_dir, "stopwords.txt"), encoding="utf-8") as f:
            data = [i.strip() for i in f.readlines()]
        return data

    def corpus_iter(self, with_label=True):
        verbose_flg = 0
        with open(os.path.join(base_dir, "cnews.train.txt"), encoding="utf-8") as f:
            for line in f:
                verbose_flg += 1
                if verbose_flg % 5000 == 0:
                    print(verbose_flg)
                label, raw_text = line.strip().split("\t")
                text = [i for i in jieba.cut(raw_text) if i not in self.stop_words and i]
                text = " ".join(text)
                if with_label:
                    yield label, text
                else:
                    yield text

    def create_vectorize(self):
        vectorize = TfidfVectorizer(min_df=100)
        corpus_iterator = self.corpus_iter(False)
        vectorize.fit(corpus_iterator)
        return vectorize

    def save_label_map(self):
        with open("../data/cnews/label_map.json", "w") as f:
            json.dump(self.label_map, f)

    @classmethod
    def get_label_map(cls):
        with open("../data/cnews/label_map.json", "r") as f:
            label_map = json.load(f)
        return label_map

    def write_tfidf_vocab(self):
        with open("../data/cnews/tf_idf_vocab.txt", mode="w", encoding="utf-8") as f:
            for word in self.vectorize.get_feature_names():
                f.write(word + "\n")

    @classmethod
    def read_vocab(self):
        word_to_idx = dict()
        with open("../data/cnews/tf_idf_vocab.txt", mode="r", encoding="utf-8") as f:
            for ix, word in enumerate(f):
                word_to_idx[word] = ix
        return word_to_idx

    def update_label_map(self, label):
        if label not in self.label_map:
            self.label_map[label] = len(self.label_map)

    def writeTFRecord(self, tf_record_file_predix, file_num=1, val_rt=None):
        corpus_iterator = self.corpus_iter(True)
        train_writers = [tf.python_io.TFRecordWriter(F"{tf_record_file_predix}_{i}.train") for i in range(file_num)]
        if val_rt:
            val_writer = tf.python_io.TFRecordWriter(F"{tf_record_file_predix}.val")
        for label, text in corpus_iterator:
            self.update_label_map(label)
            label_code = self.label_map[label]
            tfidf_vec = self.vectorize.transform([text]).toarray()[0]
            tf_example = self.create_features(tfidf_vec, label_code)

            rng = random.random()
            writer = train_writer if rng < 0.8 else val_writer
            writer.write(tf_example.SerializeToString())
        for train_writer in train_writers:
            train_writer.close()
        if val_rt:
            val_writer.close()

    def create_features(self, tfidf_vec, label):
        features = collections.OrderedDict()
        features["vec"] = self._create_float_feature(tfidf_vec)
        features["label"] = self._create_int_feature([label])
        return tf.train.Example(features=tf.train.Features(feature=features))

    def _create_int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

    def _create_float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))

    @classmethod
    def readTFRecord(cls, tf_record_file, batch_size, vocab_size):
        dataset = tf.data.TFRecordDataset(tf_record_file)
        dataset = dataset.map(cls._parse_warpper(vocab_size)).shuffle(200).batch(batch_size)
        dataset = dataset.make_one_shot_iterator().get_next()
        return dataset

    @staticmethod
    def _parse_warpper(vocab_size):
        def _parse(record):
            name_to_features = {
                "vec": tf.FixedLenFeature([vocab_size], tf.float32),
                "label": tf.FixedLenFeature([1], tf.int64)
            }
            features = tf.parse_single_example(record, name_to_features)
            return features

        return _parse


if __name__ == '__main__':
    # cnewsDataset = CnewsDataset()
    # print(cnewsDataset.voceb_size)
    # cnewsDataset.writeTFRecord("../data/cnews/train.tf_record")
    # cnewsDataset.save_label_map()
    # cnewsDataset.write_tfidf_vocab()

    VOCAB_SIZE = 10799
    dataset = CnewsDataset.readTFRecord("../data/cnews/train.tf_record.train", 64, VOCAB_SIZE)
    sess = tf.Session()
    while True:
        try:
            train_data = sess.run(dataset)
            vec, label = train_data["vec"], train_data["label"]
        except tf.errors.OutOfRangeError:
            break
