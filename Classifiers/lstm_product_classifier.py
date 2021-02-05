# For smooth importing of all the below packages (if not used, Pylance or 
# Microsoft Python Language server might cause issues)
import sys
sys.path.append("..\smartvigilance")

# Supress unimportant warnings from Transformers, Numpy and Matplotlib
import warnings
import logging
from IgnoreDependencyWarnings.ignore_warnings import IgnoreWarnings

import os
import pickle
import pandas as pd

import torch
from torch.utils.data import DataLoader
from Datasets.LSTM.lstm_custom_dataset import CustomDataset
from Datasets.LSTM.dataset_preprocessing import Preprocess

from Models.LSTM.lstm import LSTM
from Models.LSTM.lstm_trainer import LSTM_Trainer

from Evaluation.performance_evaluation_product_classifier import PerformanceEvaluation

from Visualization.visualization import Visualization

class LSTM_Product_Classifier:
    def __init__(self, dset, seed = 200):
        self.dset = dset
        self.seed = seed
        self.dset_unique_col_count = dset['device_report_product_code'].nunique()
        self.dset_label_numbers = [value for value in range(self.dset_unique_col_count)]
        self.dset_label_names = sorted(dset['device_report_product_code'].unique().tolist())

        self.text_length = 128
        self.batch_size = 64

    def prepare_dataset(self):
     
        # Ignore annoying warnings from Transformers and Matplotlib
        warnings.warn = IgnoreWarnings.warn
        IgnoreWarnings.set_global_logging_level(logging.ERROR)

        preprocess = Preprocess()

        # Label encode target attribute
        self.dset = preprocess.encode_target_attribute(self.dset)
        print(f"\nLabel Encoded Dataset:\n {self.dset}")

        # Count the frequency of each word in the data
        word_counts = preprocess.count_words(self.dset)

        # Delete infrequent words
        word_counts = preprocess.delete_less_frequent_words(word_counts, frequency_threshold=2)

        # Create an indexed word vocabulary
        word_to_idx, self.vocabulary_size = preprocess.create_word_to_idx(word_counts)

        # Save the LSTM Vocabulary for future use (during Inference/Informtion Retrieval)
        vocab_path = 'Models\LSTM\lstm_trained_model\lstm_vocab.pickle'
        with open(vocab_path, 'wb') as vocab_file:
            pickle.dump(word_to_idx, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\nLSTM Model vocabulary is saved in '{vocab_path}' for future use\n")

        max_sentence_length = max([len(t["tokenized_text"]) for _, t in self.dset.iterrows()])
        print("\nMax sentence length:", max_sentence_length)

        print("\n")
        if max_sentence_length > self.text_length:
            print("Sentence length longer than the maximum text length specified.\n"
                  "Cutting all sentences to", self.text_length)

        else:
            print("Sentence length shorter than the maximum text length specified.\n"
                  "Padding all sentences to", max_sentence_length, "\n")
            self.text_length = max_sentence_length

        # Convert the tokens in count-based embedded vector representation
        self.dset = preprocess.encode_texts(self.dset, word_to_idx, self.text_length)
        print("\n")
        print(f"Text Encoded Dataset:\n {self.dset}")

        # Split the dataframe into Train and Test
        x_train, x_valid, y_train, y_valid = preprocess.split_dataset(self.dset, self.seed)

        # Create Train and Test datasets using Torch Custom dataset
        train_ds = CustomDataset(x_train, y_train)
        valid_ds = CustomDataset(x_valid, y_valid)

        # Create Train and Test dataloaders
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)

    def train(self, epochs, learningrate):
        
        NN = LSTM_Trainer()
        perf = PerformanceEvaluation()
        viz = Visualization()

        # Initialize the LSTM model
        model = LSTM(self.vocabulary_size, NN.embedding_dimension, NN.hidden_dimension, self.dset_unique_col_count)

        # Train the model
        train_loss, val_loss = NN.train(model, self.train_loader, self.validation_loader, epochs, learningrate)

        # Testing the trained model
        targets, outputs, model_accuracy = NN.test(model, self.validation_loader)

        # Save the trained model to the disk
        model_path = "Models\LSTM\lstm_trained_model\lstm_trained_model.pt"
        torch.save(model, model_path)
        print(f"\nTrained LSTM Model saved in '{model_path}' for future use\n")

        # Compute and plot confusion matrix
        perf.overall_f1_score(targets, outputs, model_accuracy)
        perf.classification_report(targets, outputs, self.dset_label_numbers, self.dset_label_names)

        confusion_mtx = perf.compute_confusion_matrix(targets, outputs, self.dset_label_numbers)

        # Visualize confusion matrix
        viz.print_confusion_matrix(confusion_mtx, self.dset_label_names)

        # Plot training and validation curves
        viz.learning_plots(train_loss, val_loss)

# Driver program
if __name__ == "__main__":
    seed = 555
    dset = pd.read_csv(os.path.join("Datasets", "DatasetCSVFiles", "product_dataset_15k.csv"))

    lstm_classifier = LSTM_Product_Classifier(dset, seed)
    lstm_classifier.prepare_dataset()

    epoch, lr = 3, 0.01
    lstm_classifier.train(epoch, lr)