import pandas as pd

df = pd.read_csv("data/customer_judgment/train.csv")
df.info()
print(df.columns)