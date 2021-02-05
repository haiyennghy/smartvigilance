# For smooth importing of all the below packages (if not used, Pylance or 
# Microsoft Python Language server might cause issues)
import sys
sys.path.append("..\smartvigilance")

# Supress unimportant warnings from Transformers, Numpy and Matplotlib
import warnings
import logging
from IgnoreDependencyWarnings.ignore_warnings import IgnoreWarnings

import os
import numpy as np
import pandas as pd

# Torch DataImporter
import torch
from torch.utils.data import DataLoader

# Libraries for BERT custom dataset, dataset split and encoding labels
from Datasets.one_hot_label_encoder import OneHotEncode
from Datasets.BERT.bert_dataset import BERTDataset
from Datasets.BERT.bert_custom_dataset_splitter import CustomDatasetSplit

# BERT model and trainer
from Models.BERT.bert import BERTClass
from Models.BERT.bert_trainer import NeuralNetwork

# Model performance evaluation
from Evaluation.bert_performance_inherited import Performance

# Plot training curves
from Visualization.visualization import Visualization

# For real-time inference using trained model
from Models.BERT.bert_inference import BERTInference

class BERT_Product_Classifier:

    def __init__(self, dset, seed = 200, ohe_dset_path=os.path.join("Datasets", "DatasetCSVFiles", "product_dataset_onehot_15k.csv")):

        # Path to the dataset
        self.dset = dset
        self.seed = seed
        self.onehot_dataset_path = ohe_dset_path
        self.dset_unique_col_count = dset['device_report_product_code'].nunique()
        self.dset_label_numbers = [value for value in range(self.dset_unique_col_count)]
        self.dset_label_names = sorted(dset['device_report_product_code'].unique().tolist())

    def prepare_data(self):
    
        # Ignore annoying warnings from Transformers and Matplotlib
        warnings.warn = IgnoreWarnings.warn
        IgnoreWarnings.set_global_logging_level(logging.ERROR)

        # Dataset labels converted into onehot representations
        #TODO: Why do we need the onehot_dataset_path?
        # Answer: To cache the processed dataframe as a csv file for future uses (in less computational power systems)
        one_hot_df = OneHotEncode().one_hot_encoder(self.dset, self.onehot_dataset_path)

        customdataset = CustomDatasetSplit()
        train_dataset, test_dataset, validation_dataset = customdataset.create_train_val_datasets(one_hot_df, seed)

        self.NN = NeuralNetwork()

        # Datasets with all the information needed for BERT
        training_set = BERTDataset(train_dataset, self.NN.tokenizer, self.NN.max_len)
        testing_set = BERTDataset(test_dataset, self.NN.tokenizer, self.NN.max_len)
        validation_set = BERTDataset(validation_dataset, self.NN.tokenizer, self.NN.max_len)

        # Train, Test and Validation dataloaders
        self.train_loader = DataLoader(training_set, **self.NN.train_params)
        self.testing_loader = DataLoader(testing_set, **self.NN.test_params)
        self.val_loader = DataLoader(validation_set, **self.NN.validation_params)

        # Load the BERT model to GPU
        self.model = BERTClass(self.dset_unique_col_count)
        self.model.to(self.NN.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.NN.learning_rate)

    def train(self):

        # Training the model
        for epoch in range(self.NN.epochs):
            train_epoch_loss, val_epoch_loss = self.NN.train(epoch, self.train_loader, self.val_loader,
                                                             self.model, self.optimizer)


        visualization = Visualization()
        try:
            visualization.learning_plots(train_epoch_loss, val_epoch_loss)

        except:
            print("The training routine failed")

        # Testing the trained model
        outputs, targets = self.NN.testing(self.testing_loader, self.model)
        outputs = np.array(outputs) >= 0.3  #Threshold for classification

        # Generating Confusion Matrix and other performance results
        performanceevaluation = Performance()
        performanceevaluation.overall_f1_score(targets, outputs)
        performanceevaluation.classification_report(targets, outputs, self.dset_label_numbers, self.dset_label_names)
        performanceevaluation.compute_confusion_matrix(targets, outputs, self.dset_label_numbers, self.dset_label_names)

        # Using the trained model for realtime inference
        bertinference = BERTInference()
        bertinference.realtime_inference(self.model)

    def compute_feature_vector(self, dset):
        #TODO: Implement method to compute feature vectors from each text in dset
        pass


# Driver program
if __name__ == "__main__":
    seed = 555
    dset = pd.read_csv(os.path.join("Datasets", "DatasetCSVFiles", "product_dataset_15k.csv"))

    bert_classifier = BERT_Product_Classifier(dset, seed)
    bert_classifier.prepare_data()
    bert_classifier.train()
    bert_classifier.compute_feature_vector(dset)
