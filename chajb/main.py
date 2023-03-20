import pandas as pd
from Information.utils import Information_entropy, infomation_val, create_iformation_entropy, conditional_entropy, \
    union_entrpy, mutual_information, relative_entropy
from jay_plot.utils import plot_two
from metric.utils import ClassifiyMetric
from sklearn.metrics import precision_score, recall_score, f1_score


# df = pd.read_csv("data/train.csv")

# df_rslt = infomation_val(df, "area_id")
# plot_two(df_rslt, "area_id", "info_val", "prob")
# df_entrpy = create_iformation_entropy(df, df.columns)
# print(df_entrpy.sort_values(by="entropy_val"))


# for col in ["main_account_active_loan_no", "passport_flag", "Driving_flag"]:
#     val = conditional_entropy(df, x_col=col,  y_col="loan_default")
#     union_val = union_entrpy(df, x_col=col,  y_col="loan_default")
#     mutual_val = mutual_information(df, x_col="loan_default",  y_col=col)
#     KL = relative_entropy(df, x_col=col,  y_col="loan_default")
#     print(KL, col)

y_true = ["cat", "ant", "cat", "cat", "ant"]
y_pred = ["ant", "ant", "cat", "cat", "ant"]

map_ = {"cat": 0, "ant": 1}

y_true = [map_[i] for i in y_true]
y_pred = [map_[i] for i in y_pred]

print(y_true)
print(y_pred)

confusionMatrix = ClassifiyMetric(y_true, y_pred)
print(confusionMatrix.confusionmatrix)
print(confusionMatrix.precision_recall_value())
print(confusionMatrix.FPTP_rate)
print("presison_sklearn: ", precision_score(y_true, y_pred))
print("recall_sklearn: ", recall_score(y_true, y_pred))
print("f1_score_sklearn: ", f1_score(y_true, y_pred))
