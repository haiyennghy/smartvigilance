# For smooth importing of all the below packages (if not used, Pylance or 
# Microsoft Python Language server might cause issues)
import sys
sys.path.append("..\smartvigilance")

# Regular expression, string processing and efficient counting
import re
import string
from collections import Counter

# Supress unnecessary warnings from Transformers and Matplotlib
import warnings
import logging

# Tokenization, linear algebra, and easy handling of datasets
import spacy
import numpy as np
import pandas as pd

# For Performance metrics and data split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# PyTorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from IgnoreDependencyWarnings.ignore_warnings import  IgnoreWarnings
from Visualization.visualization import Visualization


class MultiColumnLabelEncoder():
    def __init__(self,columns = None):
        self.columns = columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1] # Single Embedding Vector, it's Label, it's length


class Preprocess():
    def __init__(self):
        self.language = 'en'
        self.tok = spacy.load(self.language)
        self.frequency_threshold = 2
        self.sentence_length = 85
        self.test_size = 0.2
        self.seed = 200
        self.batch_size = 2000
        self.regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')

    def read_dataset_csv(self, dataset_path):
        df = pd.read_csv(dataset_path)
        processed_df = MultiColumnLabelEncoder(columns = ['Label']).fit_transform(df)
        return processed_df
    
    def tokenize(self, text):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        nopunct = self.regex.sub(" ", text)
        return [token.text for token in self.tok.tokenizer(nopunct)]

    def count_words(self, processed_df):
        counts = Counter()
        for _, row in processed_df.iterrows():
            counts.update(self.tokenize(row['Text']))
        return counts

    def delete_less_frequenct_words(self, counts):
        print(f"Original number of words: {len(counts.keys())}")

        for word in list(counts):
            if counts[word] < self.frequency_threshold:
                del counts[word]

        print(f"After deleting less frequent words: {len(counts.keys())}")
        return counts
    
    def create_vocabulary(self, counts):
        word_to_index = {"":0, "UNK":1}
        words = ["", "UNK"]
        for word in counts:
            word_to_index[word] = len(words)
            words.append(word)
        return word_to_index, len(words)
    
    def encode_sentence(self, text, vocabulory):
        tokenized = self.tokenize(text)
        encoding_mask = np.zeros(self.sentence_length, dtype=int)
        to_be_encoded = np.array([vocabulory.get(word, vocabulory["UNK"]) for word in tokenized])
        length = min(self.sentence_length, len(to_be_encoded))
        encoding_mask[:length] = to_be_encoded[:length]
        return encoding_mask, length

    def combine_encoded_text_to_df(self, processed_df, vocabulary):
        processed_df['Encoded Text'] = processed_df['Text'].apply(lambda x: np.array(self.encode_sentence(x,vocabulary)))
        return processed_df

    def split_dataset(self, processed_df):
        x = list(processed_df['Encoded Text'])
        y = list(processed_df['Label'])
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.test_size, random_state=self.seed)
        return x_train, x_valid, y_train, y_valid
    

class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.output_dim = 9
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

    
class LSTMNeuralNetwork():
    def __init__(self):
        # Very crucial parameter. But, due to limited GPU mememory can't push this value further upto 512!!!!!
        self.embedding_dimension = 50
        self.hidden_dimension = 50
        self.epoch = 30    
        self.learningrate = 0.01

    def validation(self, model, validation_loader):
        model.eval()
        correct = 0
        total = 0
        sum_loss = 0.0
        for x, y, l in validation_loader:
            x = x.long()
            y = y.long()
            y_hat = model(x, l)
            loss = F.cross_entropy(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]
        return sum_loss/total, correct/total
    
    
    def train(self, model, train_loader, validation_loader, epoch, lr):
        train_loss_graph = []
        val_loss_graph = []
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr)
        print("\nTraining the network:\n")
        for i in range(epoch):
            model.train()
            sum_loss = 0.0
            total = 0
            for x, y, l in train_loader:
                x = x.long()
                y = y.long()
                y_pred = model(x, l)
                optimizer.zero_grad()
                loss = F.cross_entropy(y_pred, y)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]
            val_loss, val_acc = self.validation(model, validation_loader)
            train_loss_graph.append(sum_loss/total)
            val_loss_graph.append(val_loss)
            if i % 5 == 1:
                print("Train Loss: %.5f, Validation Loss: %.5f, Validation Accuracy: %.5f" % (sum_loss/total, val_loss, val_acc))
        return train_loss_graph, val_loss_graph

    
    def test(self, model, validation_loader):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        correct = 0
        total = 0
        sum_loss = 0.0
        for x, y, l in validation_loader:
            x = x.long()
            y = y.long()
            y_hat = model(x, l)
            loss = F.cross_entropy(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]
            fin_targets.extend(y)
            fin_outputs.extend(pred)
        return fin_targets, fin_outputs, correct/total


class PerformanceEvaluation():
    def __init__(self):
        self.labels_15k = [0,1,2,3,4,5,6,7,8]
        self.label_names_15k = ["CAW", "DXZ", "DZE", "FTM", "GAS", "HRY", "JAA", "MRD", "OYC"]
        self.labels_50k = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        self.label_names_50k = ["BYG", "CAW", "CCN", "DXZ", "DZE", "FTM", "GAS", "HRY", "HWC", "JAA", "LWQ", "MRD", "OYC"]
        self.average = ["micro", "macro", "weighted"]

    def overall_f1_score(self, targets, outputs, accuracy):
        print("\nTesting the network:\n")
        f1_score = []
        for f1_type in self.average:
            f1_score.append(metrics.f1_score(targets, outputs, average=f1_type))
        
        print(f"\nOverall Accuracy = {accuracy}")
        print(f"\nOverall F1 Score (Micro) = {f1_score[0]}")
        print(f"\nOverall F1 Score (Macro) = {f1_score[1]}")
        print(f"\nOverall F1 Score (Weighted) = {f1_score[2]}")
        
    def classification_report(self, targets, outputs):
        print("\nOverall Classification Report:")
        print(f"\n {metrics.classification_report(targets, outputs, labels = self.labels_15k, target_names = self.label_names_15k)}")

    def compute_confusion_matrix(self, targets, outputs):
        confusion_mtx = metrics.confusion_matrix(targets, outputs, labels = self.labels_15k)
        return confusion_mtx


# Driver program
if __name__ == "__main__":
    
    # Ignore annoying warnings from Transformers and Matplotlib
    warnings.warn = IgnoreWarnings.warn
    IgnoreWarnings.set_global_logging_level(logging.ERROR)

    # Path to the dataset
    dataset_path = "Datasets\DatasetCSVFiles\product_dataset_15k.csv"

    preprocess = Preprocess()
    NN = LSTMNeuralNetwork()
    perf = PerformanceEvaluation()
    viz = Visualization()

    # Label encoded dataset
    encoded_df = preprocess.read_dataset_csv(dataset_path)

    # Count the frequency of each word in the data
    counts = preprocess.count_words(encoded_df)

    # Delete infrequent words
    counts = preprocess.delete_less_frequenct_words(counts)

    # Create an indexed word vocabulory
    vocabulary, vocabulory_size = preprocess.create_vocabulary(counts)

    # Convert the tokens in count-based embedded vector representation
    encoded_embedded_df = preprocess.combine_encoded_text_to_df(encoded_df, vocabulary)

    # Split the dataframe into Train and Test
    x_train, x_valid, y_train, y_valid = preprocess.split_dataset(encoded_embedded_df)

    # Create Train and Test datasets using Torch Custom dataset
    train_ds = CustomDataset(x_train, y_train)
    valid_ds = CustomDataset(x_valid, y_valid)

    # Create Train and Test dataloaders
    train_loader = DataLoader(train_ds, batch_size=preprocess.batch_size, shuffle=True)
    validation_loader = DataLoader(valid_ds, batch_size=preprocess.batch_size)

    # Initialize the LSTM model
    model = LSTM(vocabulory_size, NN.embedding_dimension, NN.hidden_dimension)

    # Train the model
    train_loss, val_loss = NN.train(model, train_loader, validation_loader, NN.epoch, NN.learningrate)

    # Testing the trained model
    targets, outputs, model_accuracy = NN.test(model, validation_loader)

    # Compute and plot confusion matrix
    perf.overall_f1_score(targets, outputs, model_accuracy)
    perf.classification_report(targets, outputs)

    confusion_mtx = perf.compute_confusion_matrix(targets, outputs)

    # Visualize confusion matrix
    viz.print_confusion_matrix(confusion_mtx, perf.label_names_15k)

    # Plot training and validation curves
    viz.learning_plots(train_loss, val_loss)