import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = '#3D3D3D'


def plot_two(df, x_col, col1, col2):
    plt.subplot(2, 1, 1)
    sns.lineplot(x=x_col, y=col1, data=df, color="white")
    plt.grid()
    plt.title(col1)
    plt.subplot(2, 1, 2)
    sns.lineplot(x=x_col, y=col2, data=df, color="#CDCD00")
    plt.grid()
    plt.title(col2)
    plt.show()
