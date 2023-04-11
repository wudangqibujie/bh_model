import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv("machinelearning/RF/data/wine.txt")
    df = df[df['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)
