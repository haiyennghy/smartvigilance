# Libraries/dependencies
import sys
sys.path.append("..\smartvigilance")

import pandas as pd
import pickle
from tqdm import tqdm
from Text_Prep.maude_processing import MaudePreprocessor

class CreateTaskDataset():

    def __init__(self):
        self.maude_dataset_path = "Datasets\100000_random_entries_prod_codes.pkl"
        self.task1_dataset_file = "product_dataset.csv"
        self.sample_size = 50000

    # Saves the final dataset into a CSV file
    def dump_to_csv(self, text, labels):
        df = pd.DataFrame(data={"Text": text, "Label": labels})
        df.to_csv(self.task1_dataset_file, sep=',', index=False)
        print("Successfully saved the dataset in the desired CSV file!")

    # Use the Text_Prep pipeline to parse the raw data from MAUDE dictionary
    def parse_raw_maude_dictionary(self, sample_size):
        P = MaudePreprocessor()
        f = open(self.maude_dataset_path, 'rb')
        subset = pickle.load(f)
        final_texts = []
        label_list = []

        # Repeat for all the product entries in the dictionary
        for key in tqdm(subset):
            label = subset[key]["device"][0]['device_report_product_code']
            if "mdr_text" in subset[key]:
                per_device_texts = []

                # There are 2 text fields in "mdr_text"
                for entry in subset[key]["mdr_text"]:
                    text = entry['text']
                    processed_text = " ".join(P.pipe(text))
                    per_device_texts.append(processed_text)

            try:
                final_texts.append(" ".join(per_device_texts))
                label_list.append(label)

            except:
                print("mdr_text field not present in the dictionary")
                print("\nExiting...")
                break

            if len(label_list) >= sample_size:
                print(f"\nSuccessfully processed {sample_size} samples from the maude database")
                print("\nExiting...")
                break
        
        return final_texts, label_list

if __name__ == "__main__":
    dump_csv = CreateTaskDataset()
    text, labels = dump_csv.parse_raw_maude_dictionary(dump_csv.sample_size)
    dump_csv.dump_to_csv(text, labels)
