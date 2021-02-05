# For smooth importing of all the below packages (if not used, Pylance or 
# Microsoft Python Language server might cause issues)
import sys
sys.path.append("..\smartvigilance")

# Supress unimportant warnings from Transformers, Numpy and Matplotlib
import warnings
import logging
from IgnoreDependencyWarnings.ignore_warnings import IgnoreWarnings

import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm, naive_bayes

from Datasets.categorical_label_encoder import MultiColumnLabelEncoder
from Evaluation.performance_evaluation_product_classifier import PerformanceEvaluation
from Visualization.visualization import Visualization

class SVM_Product_Classifier:
    def __init__(self, dset, seed = 555):
        self.dset = dset
        self.seed = seed
        self.dset_testset_size = 0.3
        self.vectorizer_max_features = 8000
        self.dset_unique_col_count = dset['device_report_product_code'].nunique()
        self.dset_label_numbers = [value for value in range(self.dset_unique_col_count)]
        self.dset_label_names = sorted(dset['device_report_product_code'].unique().tolist())

    
    def prepare_data(self):

        # Ignore annoying warnings from Transformers and Matplotlib
        warnings.warn = IgnoreWarnings.warn
        IgnoreWarnings.set_global_logging_level(logging.ERROR)

        # To get consistent results (Higher seed values somehow gives better model perf results)
        np.random.seed(self.seed)

        # Splitting each row of text(which is already preprocessed and tokenized) into a iterable sublist of words
        self.dset['tokenized_text'] = [entry.split() for entry in self.dset['tokenized_text']]

        # Label encode the target variable (transform Categorical labels of string type in the data set into numerical values)
        self.dset = MultiColumnLabelEncoder(columns = ['device_report_product_code']).fit_transform(self.dset)

        # Split the data into Train and Test sets
        Train_X, Test_X, self.Train_Y, self.Test_Y = model_selection.train_test_split(self.dset['tokenized_text'].astype(str), self.dset['device_report_product_code'], test_size = self.dset_testset_size)

        # Vectorize the words by using TF-IDF Vectorizer (This step is performed to find out how important a word is in a respective document/row when it is compared it to other entities in the corpus
        Tfidf_vectorize = TfidfVectorizer(max_features = self.vectorizer_max_features)
        Tfidf_vectorize.fit(self.dset['tokenized_text'].astype(str))

        # Train X and Test X represented in terms of a TF IDF matrix which has the following structure
        # Row number, count of a single word and then it's corresponding TF-IDF value
        self.Train_X_Tfidf = Tfidf_vectorize.transform(Train_X)
        self.Test_X_Tfidf = Tfidf_vectorize.transform(Test_X)

        return self.Train_X_Tfidf, self.Test_X_Tfidf, self.Train_Y, self.Test_Y


    def fit_svm(self):
        
        # Classifier: Support Vector Classifier (SVC) from SVM
        # C is the L2 reg penalty and we are using a "Linear" kernel to handle any high dimensional non-linear features by projecting them to a high dimensional herbert space and then trying to seperate them linearly!
        self.svm_model = svm.SVC(C = 60.0, kernel = 'linear', degree = 3, gamma = 'auto')

        # Fitting the model on the vectorized training dataset
        self.svm_model.fit(self.Train_X_Tfidf, self.Train_Y)

        # predict the labels on validation dataset
        self.predictions_SVM = self.svm_model.predict(self.Test_X_Tfidf)

    
    def svm_performance_eval(self):
        perf = PerformanceEvaluation()

        # Classification report
        perf.classification_report(self.Test_Y, self.predictions_SVM, self.dset_label_numbers, self.dset_label_names)

        # Compute confusion matrix
        confusion_mtx = perf.compute_confusion_matrix(self.Test_Y, self.predictions_SVM, self.dset_label_numbers)

        viz = Visualization()

        # Visualize confusion matrix
        viz.print_confusion_matrix(confusion_mtx, self.dset_label_names)

        # Plot learning curves of SVM
        print("Please be patient, this might take a while if the input dimensions are too huge or L2 regularization value is large\n")

        viz.plot_ml_training_curves(self.Train_X_Tfidf, self.Train_Y, self.Test_X_Tfidf, self.Test_Y, self.svm_model)
        

class Naive_Bayes_Product_Classifier:
    def __init__(self, dset, seed = 555):
        self.dset = dset
        self.seed = seed


    def fit_naive_bayes(self):
        svm = SVM_Product_Classifier(self.dset, self.seed)

        train_x, test_x, train_y, test_y = svm.prepare_data()

        self.nb_model = naive_bayes.MultinomialNB()
        self.nb_model.fit(train_x,train_y)

        # predict the labels on validation dataset
        self.predictions_NB = self.nb_model.predict(test_x)

        perf = PerformanceEvaluation()

        # Classification report
        perf.classification_report(test_y, self.predictions_NB, svm.dset_label_numbers, svm.dset_label_names)

        # Compute confusion matrix
        confusion_mtx = perf.compute_confusion_matrix(test_y, self.predictions_NB, svm.dset_label_numbers)

        viz = Visualization()

        # Visualize confusion matrix
        viz.print_confusion_matrix(confusion_mtx, svm.dset_label_names)

        # Plot learning curves of SVM
        print("Please be patient, this might take a while if the input dimensions are too huge or L2 regularization value is large\n")
        viz = Visualization()
        viz.plot_ml_training_curves(train_x, train_y, test_x, test_y, self.nb_model)


# Driver program
if __name__ == "__main__":
    seed = 555
    dset = pd.read_csv(os.path.join("Datasets", "DatasetCSVFiles", "product_dataset_15k.csv"))

    svm_classifier = SVM_Product_Classifier(dset, seed)
    train_x, test_x, train_y, test_y = svm_classifier.prepare_data()
    #svm_classifier.fit_svm()
    #svm_classifier.svm_performance_eval()

    nb_classifier = Naive_Bayes_Product_Classifier(dset, seed)
    nb_classifier.fit_naive_bayes()