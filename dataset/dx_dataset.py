import pandas as pd
import random
from enum import Enum
from collections import OrderedDict
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier

CATE_COLS = [
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
NUMERIC_COLS = [
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


class FieldType(Enum):
    category = "category"
    numeric = "numeric"


class FeatureInfo:
    def __init__(self, feature_name, feature_value, feature_group_id, column_type="category", origin_feature=None):
        self.feature_name = feature_name
        self.column_type = column_type
        self.feature_group_id = feature_group_id
        self.origin_feature = origin_feature
        self.feature_value = feature_value


class FieldsInfo:
    def __init__(self, field_name, field_id, field_type):
        self.field_name = field_name
        self.field_id = field_id
        self.field_type = field_type
        self.filed_value_map = OrderedDict()

    @property
    def get_field_value_set_num(self):
        return len(self.filed_value_map)

    def update_field_info(self):
        pass


# class FeaturesInfo:
#     def __init__(self):
#         self.info = OrderedDict()
#
#     def update_info(self, field_info):


class DXDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.columns_name = self._get_columns()

    def global_shuffle(self):
        pass

    def _get_columns(self):
        with open(self.file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                break
        column_name = line.split(",")[1: -1]
        return column_name

    def read_iterator(self):
        with open(self.file_path, encoding="utf-8") as f:
            for ix, i in enumerate(f):
                if ix == 0:
                    continue
                lines = i.strip()

    def parse_line(self, line):
        lines = line.split(",")
        raw_features = lines[1: -1]
        label = int(lines[-1])
        for ix, i in enumerate(raw_features):
            is_cate_col = self.columns_name[ix] in CATE_COLS
            if is_cate_col:
                fieldsInfo = FieldsInfo(self.columns_name[ix], field_id=ix, field_type=FieldType.category.value)
            else:
                fieldsInfo = FieldsInfo(self.columns_name[ix], field_id=ix, field_type=FieldType.numeric.value)


    def writer_TFrecord(self, writer_num, tf_record_fle, file_predix=""):
        witers = [tf.python_io.TFRecordWriter(f"{tf_record_fle}_{file_predix}_{i}.tf_record") for i in range(writer_num)]








