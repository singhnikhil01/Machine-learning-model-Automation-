import pandas as pd
def list_features(dataframe):
    return list(dataframe.columns)


def delete_features(dataframe, features_to_delete: list):
    modified_dataframe = dataframe.drop(features_to_delete, axis=1)
    return modified_dataframe