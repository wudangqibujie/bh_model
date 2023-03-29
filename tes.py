from sklearn.datasets import dump_svmlight_file
import pandas as pd


df = pd.read_csv('data/spark/sample_linear_regression_data.txt', header=None, sep=' ')
dump_svmlight_file(df.iloc[:, 1:], df.iloc[:, 0], 'data/spark/sample_linear_regression_data.txt')
