import matplotlib as plt
import seaborn as sns


def draw_heatmap(review):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))
    confusion_matrixes = {
        model: review[model]["confusion matrix for test data"]
        if model != "best model"
        else None
        for model in review
    }
    del confusion_matrixes["best model"]
    row = col = 0
    for model in confusion_matrixes:
        if col in [2, 4]:
            col = 0
            row += 1
            a = sns.heatmap(
                confusion_matrixes[model],
                ax=axes[row][col],
                cmap="ocean",
                annot=True,
                cbar=False,
                annot_kws={"fontsize": 15},
            )
            a.set_title("\n" + model + "\n")
            col += 1
        else:
            a = sns.heatmap(
                confusion_matrixes[model],
                ax=axes[row][col],
                cmap="ocean",
                annot=True,
                cbar=False,
                annot_kws={"fontsize": 15},
            )
            col += 1
            a.set_title("\n" + model + "\n")
    return fig
