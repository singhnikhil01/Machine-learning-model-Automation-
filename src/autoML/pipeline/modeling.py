from autoML.pipeline.preprocessing import DataProcessor
import pandas as pd

dataa = pd.read_csv("/workspaces/Machine-learning-model-Automation-/mobile_dataset.csv")
data = DataProcessor(dataa)
print(data.describe())

'''
...show column wala cheez baki hai
ask the user to remove the features.   pass the not required cols list
'''

DataProcessor.drop_columns