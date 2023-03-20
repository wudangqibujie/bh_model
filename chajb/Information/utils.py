import pandas as pd
import math


def infomation_val(df, col):
    df_gb = df.groupby(by=col)[col].count()
    total_cnt = sum(df_gb)
    df_rt = round(df_gb / total_cnt, 5)
    df_rslt = pd.DataFrame()
    df_rslt["info_val"] = df_rt.map(lambda x :- math.log2(x))
    df_rslt["prob"] = df_rt
    return df_rslt


def Information_entropy(df):
    df["val"] = df["info_val"] * df["prob"]
    return df["val"].sum()


def create_iformation_entropy(df, columns):
    entropy_vals = []
    for col in columns:
        entropy_val = Information_entropy(infomation_val(df[[col]], col))
        entropy_vals.append(entropy_val)
    df = pd.DataFrame()
    df["cols"] = columns
    df["entropy_val"] = entropy_vals
    return df

def conditional_entropy(df, x_col, y_col):
    df_xgb = df.groupby(by=x_col)[x_col].count()
    total_x_cnt = sum(df_xgb)
    df_xrt = pd.DataFrame()
    df_xrt["prob_x"] = df_xgb.values / total_x_cnt
    df_xrt[x_col] = df_xgb.index

    x_unival = df[x_col].unique()
    entropy_vals = []
    for x_c in x_unival:
        df_y = df[df[x_col] == x_c][[y_col]]
        entropy_y_val = Information_entropy(infomation_val(df_y[[y_col]], y_col))
        entropy_vals.append(entropy_y_val)
    df_entrpy_y = pd.DataFrame()
    df_entrpy_y[x_col] = x_unival
    df_entrpy_y["condtion_entrpy"] = entropy_vals
    df_rslt = pd.merge(df_entrpy_y, df_xrt, how="left", right_on=x_col, left_on=x_col)
    df_rslt["val"] = df_rslt["condtion_entrpy"] * df_rslt["prob_x"]
    return round(df_rslt["val"].sum(), 4)

def union_entrpy(df, x_col, y_col):
    entropy_x_val = Information_entropy(infomation_val(df[[x_col]], x_col))
    conditi_entropy = conditional_entropy(df, x_col, y_col)
    return round(entropy_x_val + conditi_entropy, 5)

def mutual_information(df, x_col, y_col):
    entropy_y = Information_entropy(infomation_val(df[[y_col]], y_col))
    condi_val = conditional_entropy(df, x_col=y_col, y_col=x_col)
    return round(entropy_y - condi_val, 5)


def relative_entropy(df, x_col, y_col):
    df_gb_y = df.groupby(by=y_col)[y_col].count()
    total_cnt = sum(df_gb_y)
    df_rt_y = round(df_gb_y / total_cnt, 5)
    df_rslt_Y = pd.DataFrame()
    df_rslt_Y["info_val"] = df_rt_y.map(lambda x: - math.log2(x))
    df_rslt_Y["prob"] = df_rt_y

    df_gb_x = df.groupby(by=x_col)[x_col].count()
    total_cnt = sum(df_gb_x)
    df_rt_x = round(df_gb_x / total_cnt, 5)

    df_rslt_Y["p"] = df_rt_x.values
    df_rslt_Y["rslt"] = df_rslt_Y["p"] * df_rslt_Y["info_val"]
    return round(df_rslt_Y["rslt"].sum(), 5)









