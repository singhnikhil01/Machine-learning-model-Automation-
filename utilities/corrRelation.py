import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_and_return_correlation_matrix_image(
    dataframe,
    save_path="image.png",
    cmap="coolwarm",
    fmt=".2f",
    annot=True,
    figsize=(10, 8),
    title="Correlation Matrix Heatmap",
):
    correlation_matrix = dataframe.corr()
    sns.set(style="white")
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=fmt)
    plt.title(title)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return save_path
