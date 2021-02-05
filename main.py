from DataImporter.MAUDE.utils import load_maude_dset
from Text_Prep.report_preprocessing import tokenize_pd_maude
import pandas as pd
import os
import time

if __name__ == "__main__":
    pd.set_option('display.max_columns', 85)
    pd.set_option('display.width', 500000)
    pd.set_option('display.max_rows', 70)

    pkl_name = "subset_2.pkl"
    pkl_name = "100000_random_entries_prod_codes.pkl"
    pkl_path = os.path.join("data", "MAUDE", pkl_name)


    start = time.time()
    """Load the raw data"""
    pd_maude = load_maude_dset(pkl_path)
    print("Load:", time.time() - start)

    tokenized_pd_maude = tokenize_pd_maude(pd_maude)

    savepath = os.path.join("data", "tokenized", pkl_name)
    tokenized_pd_maude.to_pickle(savepath)


    """Load the tokenized text"""
    tokenized_texts = pd.read_pickle(savepath)
    print(tokenized_texts)



    exit(99)


