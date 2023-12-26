from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

class MachineLearningClassification:
    def __init__(self, data_pr, prediction_array=None, models=None):
        self.best_accuracy = 0
        self.prediction_array = prediction_array
        self.best_model = None
        self.best_model_object = None
        self.data = data_pr.data
        self.train_features = data_pr.train_features
        self.train_target = data_pr.train_target
        self.test_features = data_pr.test_features
        self.test_target = data_pr.test_target
        self.trained_models = []

        if models is None:
            models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(),
                      GaussianNB(), KNeighborsClassifier(), SVC()]

        self.model_evaluation_dict = {str(model).split('(')[0]: {'model_object': model} for model in models}
        self.model_prediction = {str(model).split('(')[0]: None for model in models}

    def fit(self):
        for model, dic in self.model_evaluation_dict.items():
            dic['model_object'].fit(self.train_features, self.train_target)
            self.trained_models.append(dic['model_object'])
            self.model_prediction[model] = dic['model_object'].predict(self.test_features)

    def score_test_data(self):
        for model, dic in self.model_evaluation_dict.items():
            dic['score on test data'] = dic['model_object'].score(self.test_features, self.test_target) * 100
            if dic['score on test data'] > self.best_accuracy:
                self.best_model = {'Model_obj': dic['model_object'],
                                   'Name': model,
                                   'Accuracy': dic['score on test data']}
                self.best_accuracy = dic['score on test data']

    def create_confusion_matrix(self):
        for model, dic in self.model_evaluation_dict.items():
            dic['confusion matrix for test data'] = confusion_matrix(self.test_target, self.model_prediction[model]).tolist()

    def create_f1_precision_recall(self):
        for model, dic in self.model_evaluation_dict.items():
            dic['f1 score for test data'] = f1_score(self.test_target, self.model_prediction[model], average='macro') * 100
            dic['precision for test data'] = precision_score(self.test_target, self.model_prediction[model], average='macro') * 100
            dic['recall for test data'] = recall_score(self.test_target, self.model_prediction[model], average='macro') * 100

    def evaluate(self):
        self.fit()
        self.score_test_data()
        self.create_confusion_matrix()
        self.create_f1_precision_recall()
        if isinstance(self.prediction_array, np.ndarray):
            self.model_evaluation_dict['prediction'] = self.best_model['Model_obj'].predict(
                np.array([self.prediction_array]))[0]
        for model in self.model_evaluation_dict:
            if model != 'prediction':
                del self.model_evaluation_dict[model]['model_object']
        self.best_model_object = self.best_model['Model_obj']
        del self.best_model['Model_obj']
        self.model_evaluation_dict['best model'] = self.best_model
        return self.model_evaluation_dict
