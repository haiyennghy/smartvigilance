# For smooth importing of all the below packages (if not used, Pylance or 
# Microsoft Python Language server might cause issues)
import sys
sys.path.append("..\smartvigilance")

import os
import itertools

import torch
import pickle
import pandas as pd

from Text_Prep.maude_processing import MaudePreprocessor
from Datasets.LSTM.dataset_preprocessing import Preprocess
from Evaluation.performance_evaluation_product_classifier import PerformanceEvaluation


class LSTMInference:
    def __init__(self, text_length = 128, model_path = os.path.join("Models", "LSTM", "lstm_trained_model", "lstm_trained_model.pt"), vocab_path = os.path.join("Models", "LSTM", "lstm_trained_model", "lstm_vocab.pickle")):
        self.text_length = text_length
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.retrieval_threshold = 75.00
        self.dataframe_cols = ['Document', 'Cosine_Similarity']

        # It is better creating this object in init 
        # rather than in downstream function to make code execution faster
        self.tokenize = MaudePreprocessor() 
        self.preprocess = Preprocess()
        self.perf = PerformanceEvaluation()

        if os.path.exists(self.model_path) and os.path.exists(self.vocab_path):
            self.model = torch.load(self.model_path)
            self.word_to_idx = pickle.load(open(self.vocab_path, 'rb'))
        else:
            print("\nPlease provide the correct path to the model/vocab or train the LSTM Product Classifier, save the model and try again\n")

    
    def input_text_encode(self, text):

        # Tokenize/lemmatize/... the input text
        tokenized_text = self.tokenize.pipe(text)
        #print(tokenized_text)

        # Encode the input sentence into a vector using LSTM's count-based vectorizer
        encoded_text = self.preprocess.encode_sentence(tokenized_text, self.word_to_idx, self.text_length)
        #print(encoded_text)

        return tokenized_text, encoded_text


    def predict_class(self, fully_connected_result_vector, tokenized_text):

        result = fully_connected_result_vector.tolist() #Convert tensor into a list
        result = list(itertools.chain(*result)) #Flatten the nested list into a regular list

        # Index of max value in the result vector given by FC layer of the LSTM model
        max_index = result.index(max(result))

        try:
            dset = pd.read_csv(os.path.join("Datasets", "DatasetCSVFiles", "product_dataset_15k.csv"))
            dset_label_names = sorted(dset['device_report_product_code'].unique().tolist())

            print(f"\nInput text (tokenized): {' '.join(tokenized_text)}\nPredicted class: {dset_label_names[max_index]}")

        except:
            print("\nPlease provide the input dataset file/dataframe to get the exact predicted class name, else just the class number would be displayed")
            print(f"\nInput text (tokenized): {' '.join(tokenized_text)}\nPredicted class: {max_index}")


    def lstm_inference(self, text, predict_sentence_class = False):
        
        # To deactivate Dropout and Batch Norm layers in the model
        self.model.eval()

        # Preprocess, Tokenize & Encode
        tokenized_text, encoded_text = self.input_text_encode(text)

        # Encoded text in tensor format
        encoded_text = torch.from_numpy(encoded_text)
        encoded_text = encoded_text.long().unsqueeze(0) # Manipulate the dimension (from 2 to 3, as lstm works in batches ex: [1, 64, 128], but we would have [1, 128])

        # Send the input to the pre-trained LSTM model
        fully_connected_result, feature_vector = self.model(encoded_text)

        if predict_sentence_class:
            self.predict_class(fully_connected_result, tokenized_text)

        # Return the feature vector for the input sentence (acquired from model's penaltimate layer)
        return feature_vector.tolist()


    def sentence_cosine_similarity(self, feature_vector_1, feature_vector_2):
        cos_sim = self.perf.pairwise_cosine_similarity(feature_vector_1, feature_vector_2)
        print(f"\nThe Cosine Similarity between both the input text is: {cos_sim.item() * 100}")
        return cos_sim


    def lstm_information_retrieval(self, document_list, query):
        scores = []
        query_feature_vector = self.lstm_inference(query)

        for idx, document in enumerate(document_list):
            doc_feature_vector = self.lstm_inference(document)
            cos_sim_score = self.perf.pairwise_cosine_similarity(query_feature_vector, doc_feature_vector)
            scores.append((cos_sim_score.item(), idx))  # Cosine sim score & index of the document in the input document list

        retrieved_documents = pd.DataFrame(columns = self.dataframe_cols, index = range(len(scores)))

        # Dataframe with tokenized documents & their corresponding cosine sim score with the input query
        for index in range(len(scores)):
                retrieved_documents.loc[index].Document = ' '.join(self.input_text_encode(document_list[index])[0])
                retrieved_documents.loc[index].Cosine_Similarity = scores[index][0] * 100

        print(f"\nInput Query (tokenized): {' '.join(self.input_text_encode(query)[0])}")
        print(f"\nDocuments (tokenized) and their corresponding cosine similarity with the input query:\n")

        # Sort the dataframe based on Cosine Similarity scores (descending)
        retrieved_documents.sort_values(['Cosine_Similarity'], inplace = True, ascending=False)
        print(retrieved_documents)

        return retrieved_documents


if __name__ == "__main__":

    text_1 = "Insulin Pump caused issues with the patient .."
    text_2 = "Diabetic patients need Insulin Pump"

    lstm_inf = LSTMInference()
    v1 = lstm_inf.lstm_inference(text_1, True)
    v2 = lstm_inf.lstm_inference(text_2, True)

    similarity_score = lstm_inf.sentence_cosine_similarity(v1, v2)

    # Example
    query = "Did anyone die because of a failed Insulin Pump from Abbott in the year 2015?"
    sample_documents = ["Follow Elon Musk", 
                        "In Belgium, a patient died because of a failure in his Insulin pump", 
                        "I like pizza",
                        "Snoop Dogg is the boss",
                        "Most of the Insulin pumps from Abbott have failed in the last 5 years and lead to multiple deaths"]


    print("\n")
    print("\n")

    retrieved_docs = lstm_inf.lstm_information_retrieval(sample_documents, query)
    