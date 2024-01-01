from warnings import filterwarnings
from sklearn import preprocessing, model_selection, decomposition
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from pandas import DataFrame as df
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


class DataPreprocessing:
    def __init__(self, data):
        self.data = data
        filterwarnings("ignore")
        self.objects = DataPreprocessing.initialize()
        self.input = None
        self.output = None
        self.features = list(data.columns)
        self.output_name = None
        (
            self.train_features,
            self.train_target,
            self.test_target,
            self.test_features,
            self.val_features,
            self.val_target,
        ) = (None, None, None, None, None, None)

    def drop_columns(self, columns):
        self.data.drop(columns, axis=1, inplace=True)
        if isinstance(columns, list):
            self.features = [i for i in self.features if i not in columns]
        else:
            self.features.remove(columns)

    def handle_null(self, method="drop"):
        if method == "drop":
            self.data.dropna(axis=0, inplace=True)
        elif method == "mean":
            self.data = self.data.apply(lambda x: x.fillna(x.mean()))

    def data_decription(self):
        info = self.data.info()
        description = self.data.describe()
        columns = self.data.columns
        null_data = self.data.isnull().sum()
        correlation = self.save_and_return_correlation_matrix_image()
        return info, description, columns, null_data, correlation

    @staticmethod
    def initialize():
        return {
            "Standard scaler": preprocessing.StandardScaler,
            "Min Max Scalar": preprocessing.MinMaxScaler,
            "PCA": decomposition.PCA,
            "train test split": model_selection.train_test_split,
        }

    def choose_label(self, output_name):
        self.input = self.data.drop(output_name, axis=1)
        self.output = self.data[output_name]
        self.features.remove(output_name)
        self.output_name = output_name

    def apply_count_vectorize(self, col, count_vect_obj=None):
        if count_vect_obj is None:
            self.objects["Countvec_" + col] = CountVectorizer()
            self.data[col] = self.objects["Countvec_" + col].fit_transform(
                self.data[col]
            )
        else:
            self.objects["Countvec_" + col] = count_vect_obj
            self.data[col] = self.objects["Countvec_" + col].fit_transform(
                self.data[col]
            )

    def split(self, test_percent, validation_percent=0.1, rs=42):
        (
            self.train_features,
            self.test_features,
            self.train_target,
            self.test_target,
        ) = self.objects["train test split"](
            self.input, self.output, test_size=test_percent, random_state=rs
        )

        (
            self.test_features,
            self.val_features,
            self.test_target,
            self.val_target,
        ) = self.objects["train test split"](
            self.test_features,
            self.test_target,
            test_size=validation_percent,
            random_state=rs,
        )

    def get_object_column(self):
        return [i for i in self.features if self.data[i].dtype == np.object_]

    def encode_categorical_columns(self):
        label_encoder_objects = {}
        edit_columns = self.get_object_column()
        for col in edit_columns:
            label_object = LabelEncoder()
            self.data[col] = label_object.fit_transform(self.data[col])
            label_encoder_objects[col + "_encoder_object"] = label_object
        self.objects["Label_Encoder"] = label_encoder_objects

    def change_columns(self, columns):
        self.data = self.data[columns]

    def apply_smote_data(self):
        smote_object = SMOTE()
        self.train_features, self.train_target = smote_object.fit_resample(
            self.train_features, self.train_target
        )
        self.objects["Smote object"] = smote_object

    def standardize_or_normalize(self, scale_type=None):
        if scale_type == "Standard":
            scale_object = self.objects["Standard scaler"]()
            self.train_features = df(
                data=scale_object.fit_transform(self.train_features),
                columns=self.features,
            )
            self.test_features = df(
                data=scale_object.transform(self.test_features), columns=self.features
            )
            self.val_features = df(
                data=scale_object.transform(self.val_features), columns=self.features
            )
        elif scale_type == "Normalize":
            scale_object = self.objects["Min Max Scalar"]()
            self.train_features = df(
                data=scale_object.fit_transform(self.train_features),
                columns=self.features,
            )
            self.test_features = df(
                data=scale_object.transform(self.test_features), columns=self.features
            )
            self.val_features = df(
                data=scale_object.transform(self.val_features), columns=self.features
            )

    def save_and_return_correlation_matrix_image(
        self,
        save_path="image.png",
        cmap="coolwarm",
        fmt=".2f",
        annot=True,
        figsize=(10, 8),
        title="Correlation Matrix Heatmap",
    ): 
       
        subdirectory = "assets"
        current_dir = os.getcwd()
        full_path = os.path.join(current_dir, subdirectory, save_path)
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))


        self.data = self.encode_categorical_columns()
        print(self.data)
        # correlation_matrix = self.data.corr()
        # sns.set(style="white")
        # plt.figure(figsize=figsize)
        # sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=fmt)
        # plt.title(title)
        # plt.savefig(full_path, bbox_inches="tight")
        # plt.show()
        # return full_path
