import numpy as np
import math


class Tree:
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def cal_predict_value(self, dataset):
        if self.leaf_value:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.cal_predict_value(dataset)
        else:
            return self.tree_right.cal_predict_value(dataset)

    def desc_tree(self):
        if self.tree_left and self.tree_right:
            return f"【leaf_value: {self.leaf_value}】"
        left_info = self.tree_left.desc_tree()
        right_info = self.tree_right.desc_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForest:
    def __init__(self,
                 num_tree=10,
                 max_depth=-1,
                 min_sample_split=2,
                 min_sample_leaf=1,
                 min_split_gain=0.0,
                 colsample_bytree=None,
                 subsample=0.8,
                 random_stat=None
                 ):
        self.num_tree = num_tree
        self.max_depth = max_depth if max_depth != -1 else float("inf")
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_stat = random_stat
        self.trees = None
        self.feature_importtance_ = dict()

    def fit(self, dataset, targets):
        pass

    def _parallel_build_trees(self, dataset, targets, random_stat):
        pass

    def _build_single_tree(self, dataset, targets, depth):
        pass



