# Linear algebra and array datastructures
import numpy as np

# For easy handling of datasets
import pandas as pd

# For Accuracy and F1 metrics
from sklearn import metrics

# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch dependencies
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

# Library to import pre-trained BERT model
import transformers
from transformers import BertModel, BertTokenizer, BertConfig

# Supress unnecessary warnings from Transformers and Matplotlib
import re
import warnings
import logging


class IgnoreWarnings():
    def warn(self, *args, **kwargs):
        pass

    def set_global_logging_level(self, level=logging.ERROR, prefices=[""]):
        prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
        for name in logging.root.manager.loggerDict:
            if re.match(prefix_re, name):
                logging.getLogger(name).setLevel(level)


class OneHotEncode():
    def one_hot_encoder(dataset_path, onehot_dataset_path):
        df = pd.read_csv(dataset_path)

        # One hot encoder
        one_hot_df = pd.get_dummies(df["Label"])

        # Remove labels column from original dataframe
        df = df.drop('Label',axis = 1)
        
        # Join the one hot encoded df with the original dataframe along columns
        df = df.join(one_hot_df)
        df.to_csv(onehot_dataset_path, sep=',', index=False)
        del(one_hot_df)

        df['list'] = df[df.columns[1:]].values.tolist()
        new_df = df[['Text', 'list']].copy()

        print(new_df.head())
        return new_df


class Visualization():
    def __init__(self):
        self.plot_style = 'seaborn'

    def learning_plots(self, train_loss, validation_loss, fontsize=40):
        plt.style.use(self.plot_style)
        plt.suptitle('Loss Curves', fontsize=fontsize)
        plt.plot(train_loss, label= 'Training Loss')
        plt.plot(validation_loss, color= 'orange', label= 'Validation Loss')
        plt.legend()
        plt.xlabel('Epochs/Steps')
        plt.ylabel('Loss')
        plt.show()
    
    def print_confusion_matrix(self, confusion_matrix, class_names, figsize = (10,8), fontsize=16):
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        fig = plt.figure(figsize=figsize)

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar = True)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

   
class Performance():
    def __init__(self):
        self.labels_15k = [0,1,2,3,4,5,6,7,8]
        self.label_names_15k = ["CAW", "DXZ", "DZE", "FTM", "GAS", "HRY", "JAA", "MRD", "OYC"]
        self.labels_50k = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        self.label_names_50k = ["BYG", "CAW", "CCN", "DXZ", "DZE", "FTM", "GAS", "HRY", "HWC", "JAA", "LWQ", "MRD", "OYC"]
        self.average = ["micro", "macro", "weighted"]

    def overall_evaluation(self, targets, outputs):
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score = []
        for f1_type in self.average:
            f1_score.append(metrics.f1_score(targets, outputs, average=f1_type))
        
        print(f"\nOverall Accuracy = {accuracy}")
        print(f"\nOverall F1 Score (Micro) = {f1_score[0]}")
        print(f"\nOverall F1 Score (Macro) = {f1_score[1]}")
        print(f"\nOverall F1 Score (Weighted) = {f1_score[2]}")
        print("\nOverall Classification Report:")
        print(f"\n {metrics.classification_report(targets.argmax(axis=1), outputs.argmax(axis=1), labels = self.labels_50k, target_names = self.label_names_50k)}")

    def compute_confusion_matrix(self, targets, outputs):
        targets = np.asarray(targets)
        confusion_mtx = metrics.confusion_matrix(targets.argmax(axis=1), outputs.argmax(axis=1), labels = self.labels_50k)
        Visualization.print_confusion_matrix(confusion_mtx, self.label_names_50k)


class BERTDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.Text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(comment_text, None, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation = True, return_token_type_ids=True)

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float) 
            }


class CustomDatasetSplit():
    def __init__(self):
        self.train_size = 0.7
        self.random_state = 200
        self.frac = 0.5

    def create_train_val_datasets(self, new_df):
        train_dataset = new_df.sample(frac=self.train_size, random_state=self.random_state)

        validation_dataset = new_df.drop(train_dataset.index)

        # 50% of Validation dataset and not from the entire dataset
        test_dataset = validation_dataset.sample(frac=self.frac, random_state=self.random_state)

        # Remove samples from validation dataset who are also seen in testset
        validation_dataset = validation_dataset.drop(test_dataset.index)

        # test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)

        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)
        validation_dataset = validation_dataset.reset_index(drop=True)

        print(f"\nUnique samples in Trainset: {len(train_dataset)}, Testset: {len(test_dataset)} and Validationset: {len(validation_dataset)}")

        return train_dataset, test_dataset, validation_dataset


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 13) # 13 output classes for 50,000 documents
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class NeuralNetwork():
    def __init__(self):
        # Very crucial parameter. But, due to limited GPU mememory can't push this value further upto 512!!!!!
        self.max_len = 300   
        self.train_batch_size = 32
        self.valid_batch_size = 24  # Default can be set to 8 (safe with basic GPU and low RAM)
        self.epochs = 1
        self.learning_rate = 1e-05
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = 'CUDA' if cuda.is_available() else 'CPU'
        self.train_params = {'batch_size': self.train_batch_size, 'shuffle': True, 'num_workers': 0}
        self.test_params = {'batch_size': self.valid_batch_size, 'shuffle': False, 'num_workers': 0}
        self.validation_params = {'batch_size': self.valid_batch_size,'shuffle': False,'num_workers': 0} 

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def validation(self, validation_loader, model):
        model.eval()
        val_epoch_loss = []
        with torch.no_grad():
            for _,data in enumerate(validation_loader, 0):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)

                outputs = model(ids, mask, token_type_ids)
                #optimizer.zero_grad()

                loss = self.loss_fn(outputs, targets)
                val_epoch_loss.append(loss.item())

        # print("\n-------------------------------------------------------------------")
        return val_epoch_loss

    def train(self, epoch, training_loader, validation_loader, model, optimizer):
        epoch_loss = []
        validation_epoch_loss = []
        #print("Training the model:\n")
        for _,data in enumerate(training_loader, 0):
            model.train()
            ids = data['ids'].to(self.device, dtype = torch.long)
            mask = data['mask'].to(self.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
            targets = data['targets'].to(self.device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = self.loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            # Performing validation with each step/epoch of training
            val_epoch_loss = self.validation(validation_loader, model)
            validation_loss = sum(val_epoch_loss)/len(val_epoch_loss)

            if _ % 150 == 0:
                print(f'\nEpoch: {epoch}, Step: {_}, Training Loss:  {loss.item()}, Validation Loss:  {validation_loss}')
        
            epoch_loss.append(loss.item())
            validation_epoch_loss.append(validation_loss)

        print("\nTraining Complete!")
        return epoch_loss, validation_epoch_loss

    def testing(self, testing_loader, model):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets


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
                print(f"The predicted label is: {Performance.label_names_50k[index]}")

            else:
                print("\nBye")
                break



# Driver program
if __name__ == "__main__":

    # Path to the dataset
    dataset_path = "/content/product_dataset_50k.csv"
    onehot_dataset_path = "/content/onehot_dataset_50k.csv"
    
    # Ignore annoying warnings from Transformers and Matplotlib
    warnings.warn = IgnoreWarnings.warn
    IgnoreWarnings.set_global_logging_level(logging.ERROR)

    # Dataset labels converted into onehot representations
    one_hot_df = OneHotEncode.one_hot_encoder(dataset_path, onehot_dataset_path)

    customdataset = CustomDatasetSplit()
    train_dataset, test_dataset, validation_dataset = customdataset.create_train_val_datasets(one_hot_df)

    NN = NeuralNetwork()

    # Datasets with all the information needed for BERT
    training_set = BERTDataset(train_dataset, NN.tokenizer, NN.max_len)
    testing_set = BERTDataset(test_dataset, NN.tokenizer, NN.max_len)
    validation_set = BERTDataset(validation_dataset, NN.tokenizer, NN.max_len)

    # Train, Test and Validation dataloaders
    training_loader = DataLoader(training_set, **NN.train_params)
    testing_loader = DataLoader(testing_set, **NN.test_params)
    validation_loader = DataLoader(validation_set, **NN.validation_params)

    # Load the BERT model to GPU
    model = BERTClass()
    model.to(NN.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=NN.learning_rate)

    # Training the model
    for epoch in range(NN.epochs):
        train_epoch_loss, validation_epoch_loss = NN.train(epoch, training_loader, validation_loader, model, optimizer)
    
    visualization = Visualization()
    visualization.learning_plots(train_epoch_loss, validation_epoch_loss)

    # Testing the trained model
    outputs, targets = NN.testing(testing_loader, model)
    outputs = np.array(outputs) >= 0.3  #Threshold for classification

    # Generating Confusion Matrix and other performance results
    performanceevaluation = Performance()
    performanceevaluation.overall_evaluation(targets, outputs)
    performanceevaluation.compute_confusion_matrix(targets, outputs)

    # Using the trained model for realtime inference
    bertinference = BERTInference()
    bertinference.realtime_inference(model)