import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from autoML.pipeline.preprocessing import DataProcessor
import tempfile


class Visualize:
    def __init__(self, data):
        self.data = data
        self.temp_dir = tempfile.mkdtemp()
        self.data_processor = DataProcessor(data)
        self.data_processor.encode_categorical_columns()

    def _save_plot(self, fig, image_format="png"):
        plot_file = os.path.join(self.temp_dir, f"plot.{image_format}")
        fig.savefig(plot_file, format=image_format)
        return plot_file

    def bar_plot(self, x, y, title="Bar Plot"):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=x, y=y)
        plt.title(title)
        self._save_plot(plt)

    def line_plot(self, x, y, title="Line Plot"):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x, y=y, data=self.data)
        plt.title(title)
        self._save_plot(plt)

    def scatter_plot(self, x, y, title="Scatter Plot"):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, data=self.data)
        plt.title(title)
        self._save_plot(plt)

    def histogram(self, x, title="Histogram"):
        plt.figure(figsize=(10, 6))
        sns.histplot(x=x, data=self.data, kde=True)
        plt.title(title)
        self._save_plot(plt)

    def box_plot(self, x, y, title="Box Plot"):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x, y=y, data=self.data)
        plt.title(title)
        self._save_plot(plt)

    def pie_chart(self, data, title_graph="Pie Chart"):
        fig = px.pie(data, names=data.columns[1:], values=data.columns[0], title=title_graph)
        self._save_plot(fig, image_format="png")
