from features.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
import json


class FeatureInfo:
    def __init__(self, info):
        self.info = info

    def __getitem__(self, item):
        pass


    @classmethod
    def from_config(cls, fpath):
        d_info = dict()
        infos = json.load(open(fpath))
        for feat_name, info in infos.items():
            feat_type = info["type"]
            print(info)
            info["info"]["name"] = feat_name
            assert feat_type in ("sparse", "dense", "seq")
            if feat_type == "sparse":
                d_info[feat_name] = SparseFeat(**info["info"])
            elif feat_type == "dense":
                d_info[feat_name] = DenseFeat(**info["info"])
            elif feat_type == "seq":
                info["element_info"]["name"] = feat_name
                if info["sub_type"] == "sparse":
                    info["info"].update(SparseFeat(**info["element_info"]))
                elif info["sub_type"] == "dense":
                    info["info"].update(DenseFeat(**info["element_info"]))
                else:
                    raise
                d_info[feat_name] = VarLenSparseFeat(**info["info"])
            else:
                raise
        return cls(d_info)