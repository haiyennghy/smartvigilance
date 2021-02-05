# Language Models and Text Classifiers [Feature Branch]

This branch has python and torch based implementations of neural models. For other changes to the project code, refer [Master](https://gitlab-iwi.dfki.de/smartvigilance/smartvigilance/-/tree/DFKI_master) branch.

## Environment Setup Instructions

1. Go to [root](https://gitlab-iwi.dfki.de/smartvigilance/smartvigilance/-/tree/language-model/) directory of the branch.
2. Run the following command: ``` pip install -r requirements.txt ```
3. Apart from these libraries we might need other language packs (like 'en') to be downloaded whenever it is needed.

## Extract MAUDE Product Code Dataset

1. Go to [Datasets](https://gitlab-iwi.dfki.de/smartvigilance/smartvigilance/-/blob/language-model/Datasets/) directory.
2. Run the command: ``` python create_product_classification_dataset.py ```

## Run LSTM Classifier

1. Go to [Classifiers](https://gitlab-iwi.dfki.de/smartvigilance/smartvigilance/-/tree/language-model/Classifiers) directory.
2. Run the command: ``` python lstm_product_classifier.py ```

## Run BERT Classifier

1. Go to [Classifiers](https://gitlab-iwi.dfki.de/smartvigilance/smartvigilance/-/tree/language-model/Classifiers) directory.
2. Run the command: ``` python bert_product_classifier.py ```

### Author

Akshay Joshi
