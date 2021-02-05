# For smooth importing of all the below packages (if not used, Pylance or 
# Microsoft Python Language server might cause issues)
import sys
sys.path.append("..\smartvigilance")

import torch

from Models.BERT.bert_trainer import NeuralNetwork
from Evaluation.performance_evaluation_product_classifier import PerformanceEvaluation

class BERTInference():
    def __init__(self):
        self.max_length = 300

    def get_item_tensor(self, sentence):
        inputs = NeuralNetwork.tokenizer.encode_plus(sentence, None, add_special_tokens=True, max_length= self.max_length, pad_to_max_length=True, return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)}

    def realtime_inference(self, model):
        print("\n1. Classify Sentence")
        print("\n2. Exit")
        while True:
            option = int(input("\nPlease select any option: "))
            if option == 1:
                sentence = input("\nPlease enter a description which you want to classify: ")
                tokenized_output = self.get_item_tensor(sentence)

                #print(tokenized_output["ids"].shape, tokenized_output["mask"].shape, tokenized_output["token_type_ids"].shape)

                ids = tokenized_output["ids"].to(NeuralNetwork.device, dtype = torch.long)
                mask = tokenized_output["mask"].to(NeuralNetwork.device, dtype = torch.long)
                token_type_ids = tokenized_output["token_type_ids"].to(NeuralNetwork.device, dtype = torch.long)

                output = model(ids.unsqueeze(0),mask.unsqueeze(0),token_type_ids)
                print("\n")
                output = output.tolist()
                index = output.index(max(output))
                print(f"The predicted label is: {PerformanceEvaluation.label_names_15k[index]}")

            else:
                print("\nBye")
                break