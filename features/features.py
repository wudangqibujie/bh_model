from features.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, VarLenDenseFeat
import json
import pprint


class FeatureInfo:
    def __init__(self, info):
        self.info = info

    def set_feature_idx(self, columns):
        st = 0
        for col in columns:
            feat_info = self.info[col]
            if isinstance(feat_info, SparseFeat):
                self.info["feature_idx"] = (st, st + 1)
                st += 1
            elif isinstance(feat_info, DenseFeat):
                self.info["feature_idx"] = (st, st + 1)
                st += 1
            elif isinstance(feat_info, VarLenSparseFeat):
                self.info["feature_idx"] = (st, st + feat_info.max_length)
                st += feat_info.max_length
            elif isinstance(feat_info, VarLenDenseFeat):
                self.info["feature_idx"] = (st, st + feat_info.max_length)
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