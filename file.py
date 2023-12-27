from preprocessing.dataPreprocess import DataPreprocessing
# Task-1 plot the corr of the data.
def convert_df_to_corr(data):
    data_preprocess = DataPreprocessing(data=data)
    data_preprocess.encode_categorical_columns()
    result = data_preprocess.save_and_return_correlation_matrix_image()
    return result
