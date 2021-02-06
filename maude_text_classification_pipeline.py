from Classifiers.lstm_product_classifier import LSTM_Product_Classifier
from DataImporter.MAUDE.utils import read_whole_MAUDE, load_maude_dset
from Text_Prep.report_preprocessing import tokenize_pd_maude
from DataImporter.utils import save_pd_in_chunks
from DataImporter.MAUDE.maude_dset import Maude_pd_dataset

from pandas import set_option, read_pickle
import os


def start_report_classification_preprocessing(dset_name):

    if dset_name == "100000_random_entries":
        print("Dataset:", dset_name, "\n")
        loadpath = os.path.join("data", "MAUDE", "100000_random_entries_prod_codes.pkl")
        savepath = os.path.join("data", "tokenized", "100000_random_entries_prod_codes.pkl")


    elif dset_name == "subset_2":
        print("Dataset:", dset_name, "\n")
        loadpath = os.path.join("data", "MAUDE", "subset_2.pkl")
        savepath = os.path.join("data", "tokenized", "subset_2.pkl")


    elif dset_name == "whole_maude":
        pd_whole_dset = read_whole_MAUDE(version="raw")
        pd_whole_maude = tokenize_pd_maude(pd_whole_dset)
        save_pd_in_chunks(pd_whole_maude, os.path.join("data", "tokenized", "MAUDE"), 1000000)

        return pd_whole_maude

    else:
        raise ValueError("dset_name must be one of [100000_random_entries, whole_maude]. Got", dset_name)

    dset_maude = load_maude_dset(loadpath)
    dset_maude_tokenized = tokenize_pd_maude(dset_maude)

    dset_maude_tokenized.dataset.to_pickle(savepath)
    print("Tokenized texts saved")

    return dset_maude_tokenized

def start_report_classification_training(dset_name):
    print("Start report classification using MAUDE\n")

    if dset_name == "100000_random_entries":
        print("Dataset:", dset_name, "\n")
        loadpath = os.path.join("data", "tokenized", "100000_random_entries_prod_codes.pkl")
        pd_maude = read_pickle(loadpath)
        print(pd_maude)

    elif dset_name == "subset_2":
        print("Dataset:", dset_name, "\n")
        loadpath = os.path.join("data", "tokenized", "subset_2.pkl")
        pd_maude = read_pickle(loadpath)
        print(pd_maude)

    elif dset_name == "whole_maude":
        pd_maude = read_whole_MAUDE(version="tokenized")

    else:
        raise ValueError("dset_name must be one of [100000_random_entries, whole_maude]. Got", dset_name)

    pd_dset_tokenized = Maude_pd_dataset(pd_maude)
    pd_dset_tokenized.unpack_device_column()
    print(pd_dset_tokenized.dataset)
    print(pd_dset_tokenized.dataset[["tokenized_text", "device_report_product_code"]])

    exit(99)

    train_LSTM(pd_dset_tokenized)

def train_LSTM(pd_dset_tokenized):

    """Train the LSTM classifier"""
    lstm_classifier = LSTM_Product_Classifier(pd_dset_tokenized.dataset)
    lstm_classifier.prepare_dataset()
    lstm_classifier.train(epochs=3, learningrate=0.005)

def train_BERT(pd_dset_tokenized):
    pass

def train_SVM(pd_dset_tokenized):
    pass



if __name__ == "__main__":
    set_option('display.max_columns', 95)
    set_option('display.width', 500000)
    set_option('display.max_rows', 70)


    dset = "100000_random_entries"
    #dset = "subset_2"

    start_report_classification_preprocessing(dset)
    start_report_classification_training(dset)



