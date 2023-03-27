import pandas as pd
import random
import json
import collections
import tensorflow as tf

cate_cols = [
    "地理区域",
    "是否双频",
    "是否翻新机",
    "手机网络功能",
    "婚姻状况",
    "信息库匹配",
    "信用卡指示器",
    "新手机用户",
    "信用等级代码",
    "账户消费限额",
]

numeric_cols = [
    "当前手机价格",
    "家庭成人人数",
    "预计收入",
    "当前设备使用天数",
    "在职总月数",
    "家庭中唯一订阅者的数量",
    "家庭活跃用户数",
    "平均月费用",
    "每月平均使用分钟数",
    "平均超额使用分钟数",
    "平均超额费用",
    "平均语音费用",
    "数据超载的平均费用",
    "平均漫游呼叫数",
    "当月使用分钟数与前三个月平均值的百分比变化",
    "当月费用与前三个月平均值的百分比变化",
    "平均掉线语音呼叫数",
    "平均丢弃数据呼叫数",
    "平均占线语音呼叫数",
    "平均占线数据调用次数",
    "平均未接语音呼叫数",
    "未应答数据呼叫的平均次数",
    "尝试拨打的平均语音呼叫次数",
    "尝试数据调用的平均数",
    "平均接听语音电话数",
    "平均完成的语音呼叫数",
    "完成数据调用的平均数",
    "平均客户服务电话次数",
    "使用客户服务电话的平均分钟数",
    "一分钟内的平均呼入电话数",
    "平均三通电话数",
    "已完成语音通话的平均使用分钟数",
    "平均呼入和呼出高峰语音呼叫数",
    "平均峰值数据调用次数",
    "使用高峰语音通话的平均不完整分钟数",
    "平均非高峰语音呼叫数",
    "非高峰数据呼叫的平均数量",
    "平均掉线或占线呼叫数",
    "平均尝试调用次数",
    "平均已完成呼叫数",
    "平均呼叫转移呼叫数",
    "平均呼叫等待呼叫数",
    "客户生命周期内的总通话次数",
    "客户生命周期内的总使用分钟数",
    "客户生命周期内的总费用",
    "计费调整后的总费用",
    "计费调整后的总分钟数",
    "计费调整后的呼叫总数",
    "客户生命周期内平均月费用",
    "客户生命周期内的平均每月使用分钟数",
    "客户整个生命周期内的平均每月通话次数",
    "过去三个月的平均每月使用分钟数",
    "过去三个月的平均每月通话次数",
    "过去三个月的平均月费用",
    "过去六个月的平均每月使用分钟数",
    "过去六个月的平均每月通话次数",
    "过去六个月的平均月费用",
]

df = pd.read_csv("../data/train.csv")

df_num = df[numeric_cols]
df_label = df[["是否流失"]]
df_cate = pd.get_dummies(df[cate_cols].astype("str"))

df_T = pd.concat([df_label, df_num, df_cate], axis=1)
print(df_T.shape)


class DataSet:
    def __init__(self, tf_file, if_onehot=False):
        self.tf_file = tf_file
        self.if_onehot = if_onehot
        self.df = pd.read_csv("../data/train.csv")
        self.cate_cols = ["地理区域",
                          "是否双频",
                          "是否翻新机",
                          "手机网络功能",
                          "婚姻状况",
                          "信息库匹配",
                          "信用卡指示器",
                          "新手机用户",
                          "信用等级代码",
                          "账户消费限额", ]
        self.numeric_cols = ["当前手机价格",
                             "家庭成人人数",
                             "预计收入",
                             "当前设备使用天数",
                             "在职总月数",
                             "家庭中唯一订阅者的数量",
                             "家庭活跃用户数",
                             "平均月费用",
                             "每月平均使用分钟数",
                             "平均超额使用分钟数",
                             "平均超额费用",
                             "平均语音费用",
                             "数据超载的平均费用",
                             "平均漫游呼叫数",
                             "当月使用分钟数与前三个月平均值的百分比变化",
                             "当月费用与前三个月平均值的百分比变化",
                             "平均掉线语音呼叫数",
                             "平均丢弃数据呼叫数",
                             "平均占线语音呼叫数",
                             "平均占线数据调用次数",
                             "平均未接语音呼叫数",
                             "未应答数据呼叫的平均次数",
                             "尝试拨打的平均语音呼叫次数",
                             "尝试数据调用的平均数",
                             "平均接听语音电话数",
                             "平均完成的语音呼叫数",
                             "完成数据调用的平均数",
                             "平均客户服务电话次数",
                             "使用客户服务电话的平均分钟数",
                             "一分钟内的平均呼入电话数",
                             "平均三通电话数",
                             "已完成语音通话的平均使用分钟数",
                             "平均呼入和呼出高峰语音呼叫数",
                             "平均峰值数据调用次数",
                             "使用高峰语音通话的平均不完整分钟数",
                             "平均非高峰语音呼叫数",
                             "非高峰数据呼叫的平均数量",
                             "平均掉线或占线呼叫数",
                             "平均尝试调用次数",
                             "平均已完成呼叫数",
                             "平均呼叫转移呼叫数",
                             "平均呼叫等待呼叫数",
                             "客户生命周期内的总通话次数",
                             "客户生命周期内的总使用分钟数",
                             "客户生命周期内的总费用",
                             "计费调整后的总费用",
                             "计费调整后的总分钟数",
                             "计费调整后的呼叫总数",
                             "客户生命周期内平均月费用",
                             "客户生命周期内的平均每月使用分钟数",
                             "客户整个生命周期内的平均每月通话次数",
                             "过去三个月的平均每月使用分钟数",
                             "过去三个月的平均每月通话次数",
                             "过去三个月的平均月费用",
                             "过去六个月的平均每月使用分钟数",
                             "过去六个月的平均每月通话次数",
                             "过去六个月的平均月费用", ]
        self.label = "是否流失"
        self.cate_fea_num = len(self.cate_cols)
        self.numeric_fea_num = len(self.numeric_cols)
        self.cate_max_id = None

    def onehot(self):
        return pd.get_dummies(self.df[self.cate_cols].astype("str"))

    def write_cols_info(self, file_name, cols):
        with open(file_name, encoding="utf-8", mode="w") as f:
            for c in cols:
                f.write(c + "\n")

    def reset_cate_id(self):
        max_flg = 0
        for cate_col in self.cate_cols:
            uni_vals = self.df[cate_col].unique()
            re_map = {uni_vals[i]: i for i in range(len(uni_vals))}
            self.df[cate_col] = self.df[cate_col].map(re_map)
            max_flg = 0 if max_flg == 0 else max_flg
            self.df[cate_col] = self.df[cate_col] + max_flg
            max_flg = self.df[cate_col].max()
        self.cate_max_id = max_flg

    def _create_int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

    def _create_float_feature(self, values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

    def create_example(self, numeric_feature, cate_features, label):
        features = collections.OrderedDict()
        features["numeric_feature"] = self._create_float_feature(numeric_feature)
        features["cate_features"] = self._create_int_feature(cate_features)
        features["label"] = self._create_int_feature([label])
        return tf.train.Example(features=tf.train.Features(feature=features))

    def write_TFrecord(self):
        self.reset_cate_id()
        data = self.df[[self.label] + self.numeric_cols + self.cate_cols].values
        train_writer = tf.python_io.TFRecordWriter("train" + self.tf_file)
        val_write = tf.python_io.TFRecordWriter("val" + self.tf_file)
        feature_cap = len(self.numeric_cols)
        for i in data:
            label = i[0]
            numeric_features = i[1: 1 + feature_cap]
            cate_features = i[feature_cap + 1:]
            tf_example = self.create_example(numeric_features, cate_features, label)
            rng = random.random()
            writer = train_writer if rng < 0.8 else val_write
            writer.write(tf_example.SerializeToString())
        train_writer.close()
        val_write.close()

    def _parse(self, record):
        name_to_features = {
            "numeric_feature": tf.FixedLenFeature([self.numeric_fea_num], tf.float32),
            "cate_features": tf.FixedLenFeature([self.cate_fea_num], tf.int64),
            "label": tf.FixedLenFeature([1], tf.int64)
        }
        features = tf.parse_single_example(record, name_to_features)
        return features

    def read_TFRecord(self, tf_file, batch_size=64):
        dataset = tf.data.TFRecordDataset(tf_file)
        dataset = dataset.map(self._parse).shuffle(100).batch(batch_size)
        dataset = dataset.make_one_shot_iterator().get_next()
        return dataset


# if __name__ == '__main__':
    # dataSet = DataSet("DY.tf_dataset")
    # dataSet.reset_cate_id()
    # print(dataSet.cate_max_id)
    # dataSet.write_TFrecord()
    # tf_dataset = dataSet.read_TFRecord("trainDY.tf_dataset")
    # print(dataSet.cate_fea_num, dataSet.numeric_fea_num)
    # sess = tf.Session()
    # while True:
    #     try:
    #         rs = sess.run(tf_dataset)
    #         print(rs)
    #     except tf.errors.OutOfRangeError:
    #         break

    # from pyspark.sql.functions import when
    #
    # df = df.withColumn('category', when(df.age < 18, 'child').otherwise('adult'))