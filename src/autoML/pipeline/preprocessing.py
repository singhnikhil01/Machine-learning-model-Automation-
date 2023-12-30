import json
from autoML.preprocessing.dataPreprocess import DataPreprocessing

class DataProcessor:
    def __init__(self, data):
        self.data_preprocess = DataPreprocessing(data=data)

    def convert_df_to_corr(self):
        self.data_preprocess.encode_categorical_columns()
        result = self.data_preprocess.save_and_return_correlation_matrix_image()
        return result

    def drop_columns(self, columns=None):
        if columns is None:
            columns = []
        self.data_preprocess.drop_columns(columns=columns)

    def handle_null(self, method="drop"):
        self.data_preprocess.handle_null(method=method)

    def split_data(self, test_percent=0.2, validation_percent=0.1, rs=42):
        self.data_preprocess.split(test_percent, validation_percent, rs)

    def choose_label(self, label: str):
        self.data_preprocess.choose_label(label)

    def change_columns(self, columns):
        self.data_preprocess.change_columns(columns=columns)

    def describe(self):
        info, description, columns, null_data, correlation = self.data_preprocess.data_decription()
        data_description_dict = {
            "info": info.to_dict(),
            "description": description.to_dict(),
            "columns": columns,
            "null_data": null_data.to_dict(),
            "correlation": correlation,
        }
        data_description_json = json.dumps(data_description_dict, indent=4)
        return data_description_json

    def standardize_or_normalize(self, scale_type=None):
        self.data_preprocess.standardize_or_normalize(scale_type=scale_type)

    def apply_smote_data(self):
        self.data_preprocess.apply_smote_data()

    def encode_categorical_columns(self):
        self.data_preprocess.encode_categorical_columns()

    def apply_count_vectorize(self, col, count_vect_obj=None):
        self.data_preprocess.apply_count_vectorize(col, count_vect_obj=count_vect_obj)

    def initialize(self):
        self.data_preprocess.initialize()


