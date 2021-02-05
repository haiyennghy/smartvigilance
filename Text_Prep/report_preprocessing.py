import math
import time

import numpy as np
import pandas as pd

from Text_Prep.maude_processing import MaudePreprocessor
from DataImporter.MAUDE.maude_dset import Maude_pd_dataset


"""
Scripts to preprocess the texts in the data, i.e. texts in MAUDE reports or Medline Papers
"""



def tokenize_pd_maude(maude_pd_dataset:Maude_pd_dataset):

    """Get all texts from the MAUDE entries, i.e. unpack the mdr_text field to be one row per text"""
    start = time.time()
    maude_pd_dataset.explode()
    print("Explode:", time.time() - start)
    maude_pd_dataset.unpack_mdr_text_column()
    maude_pd_dataset.dataset.dropna(subset=["text"], inplace=True)
    print(maude_pd_dataset.dataset)

    """Tokenize the dataset"""
    """takes some time. Afterwards, save as .pkl"""
    pd_tokenized = apply_tokenization_on_dataset(maude_pd_dataset.dataset)

    return Maude_pd_dataset(pd_tokenized)



def apply_tokenization_on_dataset(pd_texts):

    pd_texts["tokenized_text"] = pd.NA

    maude_prep = MaudePreprocessor()

    print("\n\nNumber of texts to process:", len(pd_texts))
    start = time.time()


    for i, (idx, row) in enumerate(pd_texts.iterrows()):

        try:
            text_pre = maude_prep.pipe(row["text"], only_tokens=True)
            pd_texts.at[idx, "tokenized_text"] = text_pre
        except:
            """Most of the exceptions come from nan values, if not print the line for debugging"""
            print(row["text"])
            pd_texts.at[idx, "tokenized_text"] = text_pre

            #pd_texts.at[idx, "tokenized_text"] = text_pre

            #if type(row["mdr_text"]) == float:
            #    if not math.isnan(row["mdr_text"]):
            #        print(row["mdr_text"])
            #else:
            #    print(row["mdr_text"])

        if i > 2 and i % 5000 == 0:
            print(i)
            #print(time.time() - start)
            start = time.time()

    """Remove the rows with empty texts"""
    pd_texts.dropna(subset=["tokenized_text"], inplace=True)
    print(pd_texts)

    print("Number of unique tokens: ", len(pd_texts["tokenized_text"].explode().unique()))

    print("Finished")

    return pd_texts



def tokenize_dataset_in_batches(pd_texts):
    """
    Process texts in batches which should be faster according to the authors.
    However, its slower and the returned batch cannot be split back in the original texts.

    :param pd_texts:
    :return:
    """

    raise NotImplementedError()

    pd_texts["tokenized text"] = ""

    print(pd_texts)

    maude_prep = MaudePreprocessor()

    print(len(pd_texts))
    start = time.time()

    batch = ""
    for i, (idx, row) in enumerate(pd_texts.iterrows()):

        try:
            text = row["mdr_text"]["text"]
        except:
            text = "NAN"

        """Text is sometimes nan which throws error when tokenizing. Use string NAN instead"""
        if text != np.nan:
            batch += text + "\n\n"
        else:
            print(text)
            batch += "NAN" + "\n\n"

        if (i+1)%1000==0:
            #print(batch)

            try:
                prep = maude_prep.pipe(batch, only_tokens=True)

                #print(prep)

                for tokenized in prep:
                    #print(tokenized)
                    if tokenized == "NAN":
                        continue

                    #pd_texts.at[idx, "tokenized text"] = prep
            except:
                print(batch)
                exit(99)

            print(i)
            print(time.time() - start)
            start = time.time()
            batch = ""

    print(pd_texts)

    print("Finished")


    return pd_texts