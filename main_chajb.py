import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
pd.set_option("display.max_columns", None)


df = pd.read_csv("chajb/data/train.csv")
y = df["loan_default"]
X = df.drop(columns=["customer_id", "loan_default", "outstanding_disburse_ratio"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape)
# print(df.describe())
model = LogisticRegression()
model.fit(X_train, y_train)

y_train_score = model.predict_proba(X_train)[:, 1]
auc_train = metrics.roc_auc_score(y_train, y_train_score)
y_test_score = model.predict_proba(X_test)[:, 1]
auc_test = metrics.roc_auc_score(y_test, y_test_score)
fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_train_score)
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_test_score)
ks_train = abs(fpr_train - tpr_train).max()
ks_test = abs(fpr_test - tpr_test).max()
print(f"train_auc: {round(auc_train, 4)}, train_ks: {round(ks_train, 4)}")
print(f"test_auc: {round(auc_test, 4)}, test_ks: {round(ks_test, 4)}")



