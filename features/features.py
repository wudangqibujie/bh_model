from features.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, VarLenDenseFeat
import json
import pprint


class FeatureInfo:
    def __init__(self, info):
        self.info = info

    @property
    def total_sparse_feats(self):
        return list(filter(lambda x: isinstance(x, SparseFeat), self.info.values()))

    @property
    def total_dense_feats(self):
        return list(filter(lambda x: isinstance(x, DenseFeat), self.info.values()))

    @property
    def total_sequcen_feat(self):
        return list(filter(lambda x: isinstance(x, VarLenSparseFeat) or isinstance(x, VarLenDenseFeat), self.info.values()))

    def set_feature_idx(self, columns):
        st = 0
        for col in columns:
            feat_info = self.info[col]
            if isinstance(feat_info, SparseFeat):
                feat_info.feature_idx = (st, st + 1)
                # self.info["feature_idx"] = (st, st + 1)
                st += 1
            elif isinstance(feat_info, DenseFeat):
                feat_info.feature_idx = (st, st + 1)
                # self.info["feature_idx"] = (st, st + 1)
                st += 1
            elif isinstance(feat_info, VarLenSparseFeat):
                feat_info.feature_idx = (st, st + feat_info.max_length)
                # self.info["feature_idx"] = (st, st + feat_info.max_length)
                st += feat_info.max_length
            elif isinstance(feat_info, VarLenDenseFeat):
                feat_info.feature_idx = (st, st + feat_info.max_length)
                # self.info["feature_idx"] = (st, st + feat_info.max_length)
                st += feat_info.max_length
            else:
                raise

    def __str__(self):
        pass

    @classmethod
    def from_config(cls, fpath):
        d_info = dict()
        infos = json.load(open(fpath))
        for feat_name, info in infos.items():
            feat_type = info["type"]
            info["info"]["name"] = feat_name
            assert feat_type in ("sparse", "dense", "sequence")
            if feat_type == "sparse":
                d_info[feat_name] = SparseFeat(**info["info"])
            elif feat_type == "dense":
                d_info[feat_name] = DenseFeat(**info["info"])
            elif feat_type == "sequence":
                info["element_info"]["name"] = feat_name
                if info["sub_type"] == "sparse":
                    info["info"].update({"sparsefeat": SparseFeat(**info["element_info"])})
                    d_info[feat_name] = VarLenSparseFeat(**info["info"])
                elif info["sub_type"] == "dense":
                    info["info"].update({"densefeat": SparseFeat(**info["element_info"])})
                    d_info[feat_name] = VarLenDenseFeat(**info["info"])
                else:
                    raise
            else:
                raise
        return cls(d_info)