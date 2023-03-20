import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


class ClassifiyMetric:
    def __init__(self, y_true, y_pred, labels=None):
        self.labels = labels
        self._matrix = confusion_matrix(y_true, y_pred, labels)
        self._warrper_to_pandas()

    def _warrper_to_pandas(self):
        item_num = self._matrix.shape[0]
        true_cols = self.labels if self.labels else [i for i in range(item_num)]
        self._true_cols = [f"true_{i}" for i in true_cols]
        headers = self.labels if self.labels else [i for i in range(item_num)]
        self._headers = [f"pred_{i}" for i in headers]
        self._matrix_df = pd.DataFrame(data=self._matrix, columns=self._headers, index=self._true_cols)

    @property
    def confusionmatrix(self):
        return self._matrix_df

    def precision_recall_value(self, label=None):
        assert self._matrix_df.shape[0] == 2
        pred_sum = np.sum(self._matrix, axis=0)
        true_sum = np.sum(self._matrix, axis=1)
        cate_idx_pred = self._headers.index(f"pred_{label}") if label else 1
        cate_idx_true = self._true_cols.index(f"true_{label}") if label else 1
        true_val = self._matrix[cate_idx_true][cate_idx_pred]
        precision = round(true_val / pred_sum[cate_idx_pred], 3)
        recall = round(true_val / true_sum[cate_idx_true], 3)
        f1_score = round(2 * (precision * recall / (precision + recall)), 3)
        return {"precision": precision, "recall": recall, "f1_score": f1_score}

    @property
    def FPTP_rate(self):
        assert self._matrix_df.shape[0] == 2
        true_sum = np.sum(self._matrix, axis=1)
        tp_num = self._matrix[1][1]
        fp_num = self._matrix[0][1]
        tp_rate = round(tp_num / true_sum[1], 3)
        fp_rate = round(fp_num / true_sum[0], 3)
        return {"TP_rate": tp_rate, "FP_rate": fp_rate}




