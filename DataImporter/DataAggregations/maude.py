import os
import spacy
import numpy as np
import sys
sys.path.append("..")
from collections import Counter
import matplotlib.pyplot as plt
#import smartvigilance.DataImporter.utils as utils
import smartvigilance.DataImporter.utils as utils
from smartvigilance.DataImporter.utils import open_json, save_dict_as_pkl, load_pkl, save_pd_in_chunks
import pandas as pd


def get_text_from_json(json):
    if "mdr_text" in list(json.keys()):
        try:
            mdr_text = json["mdr_text"]
            if len(mdr_text) > 0:
                texts = []
                for i, text in enumerate(mdr_text):
                    if "text" in list(text.keys()):
                        texts.append(text["text"])

                return texts
        except Exception as e:
            print(e)
            print(json["mdr_text"])
            exit(99)


def print_entry(entry):
    for key, value in entry.items():
        print(key, ":", value)

def collect_all_text_entries():
    root_path = os.path.join("MAUDE/")

    all_texts = {}
    max_texts = 0

    for path, subdir, files in os.walk(root_path):
        #print(path)
        for file in os.listdir(path):
            if file == "download.json":
                continue

            if ".json" in file:
                print(os.path.join(path, file))

                data = open_json(os.path.join(path, file))

                entries = data["results"]
                print("Entries: \t", len(entries))

                texts = []
                for entry in entries:
                    text = get_text_from_json(entry)
                    if text is not None:
                        max_texts = max(max_texts, len(text))
                        texts.extend(text)

                print("Texts: \t\t", len(texts))

                all_texts[os.path.join(path, file)] = texts

    print(len(list(all_texts.keys())))
    print(max_texts)

    save_dict_as_pkl(all_texts, "MAUDE_all_texts/MAUDE_all_texts")










def collect_all_entries():
    maude_raw_data_root_path = os.path.join("..", "..", "data", "MAUDE")

    all_entries = pd.DataFrame()

    for path, subdir, files in os.walk(maude_raw_data_root_path):
        # print(path)
        for file in os.listdir(path):
            if file == "download.json":
                continue

            if ".json" in file:
                print(os.path.join(path, file))

                data = open_json(os.path.join(path, file))

                entries = data["results"]
                print("Entries: \t", len(entries))
                all_entries = all_entries.append(entries)

            print("All entries:", len(all_entries), "\n")

    save_pd_in_chunks(all_entries, os.path.join("..", "..", "data", "MAUDE", "all_entries"), 100000)
    #all_entries.to_pickle(os.path.join("..", "..", "data", "MAUDE", "all_entries.pkl"))
    #save_dict_as_pkl(all_entries, os.path.join("..", "..", "data", "MAUDE", "all_entries.pkl"))
















def collect_all_entries_by_product_code(prod_code_list):
    root_path = os.path.join("MAUDE/")

    all_entries = {}

    for path, subdir, files in os.walk(root_path):
        # print(path)
        for file in os.listdir(path):
            if file == "download.json":
                continue

            if ".json" in file:
                print(os.path.join(path, file))

                data = open_json(os.path.join(path, file))

                entries = data["results"]
                print("Entries: \t", len(entries))
                for entry in entries:
                    #print_entry(entry)

                    if "device" in entry.keys():
                        try:
                            devices = entry["device"]
                        except:
                            print_entry(entry)
                            print("empty\n\n")

                        else:
                            """Execute if no exception was risen"""
                            if len(devices) == 1:
                                if devices[0]["device_report_product_code"] in prod_code_list:
                                    all_entries[entry["mdr_report_key"]] = entry

                            else:
                                for i, device in enumerate(devices):
                                    if device["device_report_product_code"] in prod_code_list:
                                        all_entries[entry["mdr_report_key"]] = entry
                                        break

            print(len(all_entries))
            if len(all_entries) > 2000000:
                raise Exception("Entries too big", len(all_entries))




    print(len(list(all_entries.keys())))

    save_dict_as_pkl(all_entries, "MAUDE/all_entries_prod_codes")


def split_maude_all_texts_in_subdicts(path):
    all_texts = load_pkl(path)

    root = "MAUDE_all_texts"

    for name, texts in all_texts.items():
        name = name.split("/")
        name = name[1] + "_" + name[2]
        name = name[:-5]
        print(name)
        new_dict = dict({name: texts})
        save_dict_as_pkl(new_dict, os.path.join(root, name))


def collect_texts_by_year(path, year):
    all_texts = load_pkl(path)

    all_texts_year = {year: []}

    for name, texts in all_texts.items():
        if year in name:
            all_texts_year[year].extend(texts)

    save_dict_as_pkl(all_texts_year, "MAUDE_all_texts/MAUDE_all_texts_" + year)

def tokenize_texts(texts):
    from spacy.attrs import ORTH
    nlp = spacy.load("en_core_web_sm")

    b4 = [{ORTH: "(B)(4)"}]
    b6 = [{ORTH: "(B)(6)"}]
    nlp.tokenizer.add_special_case("(B)(4)", b4)
    nlp.tokenizer.add_special_case("(B)(6)", b6)


    tokenized_texts = []
    for i, text in enumerate(texts):
        tokens = nlp(text)
        tokens = [str(t) for t in tokens]
        tokenized_texts.append(tokens)

    return tokenized_texts


def tokenzize_all_texts(all_texts, year=""):

    num_docs = 0
    all_texts_tokenized = {}

    for doc, texts in all_texts.items():
        if year in doc:
            all_texts_tokenized[doc] = tokenize_texts(texts)

        print(doc)
        num_docs += 1

    print("Tokenization finished")

    return all_texts_tokenized

    if year != "":
        save_dict(all_texts_tokenized, "MAUDE_all_texts/MAUDE_all_texts_" + year + "_tokenized")
    else:
        save_dict(all_texts_tokenized, "MAUDE_all_texts/MAUDE_all_texts_tokenized")




def aggregate_texts_to_list(path):

    dict = load_pkl(path)
    all_texts = []
    for name, texts in dict.items():
        all_texts.extend(texts)

    return all_texts

def most_common_words(word_list):
    word_count = Counter(word_list)
    return word_count

#collect_all_text_entries()

path_MAUDE_all_texts = "MAUDE_all_texts/MAUDE_all_texts"
path_MAUDE_subdirs = "MAUDE_all_texts"

#split_maude_all_texts_in_subdicts(path_maude_all_texts)
#compute_average_textlen(path_MAUDE_subdirs)

#all_texts = load_dict("MAUDE_all_texts/MAUDE_all_texts")
#print("Number of texts: ", sum(len(i) for i in list(all_texts.values())))
#tokenzize_all_texts(all_texts)
#tokenzize_all_texts(all_texts, "2019")

if __name__ == "__main__":

    #collect_all_entries_by_product_code(["OYC", "DZE", "JAA", "CAW", "FTM", "HRY", "DXZ", "GAS", "MRD", "CCN"])
    #exit(99)

    collect_all_entries()
    exit(99)

    all_texts = load_pkl("MAUDE_all_texts/MAUDE_all_texts")
    print(all_texts.keys())
    """
    all_texts = {'MAUDE/2005q4/device-event-0001-of-0001.json': all_texts['MAUDE/2005q4/device-event-0001-of-0001.json'],
                'MAUDE/1997q1/device-event-0001-of-0001.json': all_texts['MAUDE/1997q1/device-event-0001-of-0001.json'],
                'MAUDE/2014q2/device-event-0001-of-0002.json': all_texts['MAUDE/2014q2/device-event-0001-of-0002.json']}
    """
    print("Number of texts: ", sum(len(i) for i in list(all_texts.values())))
    all_texts_tokenized = utils.parallelized_processing(np.array(list(all_texts.items())), tokenzize_all_texts, 8)
    print(len(all_texts_tokenized))

    save_dict_as_pkl(all_texts_tokenized, "MAUDE_all_texts/MAUDE_all_texts_tokenized")
    #tokenzize_all_texts(all_texts)


    path_MAUDE_texts_tokenized = "MAUDE_all_texts/MAUDE_all_texts_tokenized"
    texts = aggregate_texts_to_list(path_MAUDE_texts_tokenized)
    print(texts[0:10])
    print(len(texts))

    text_lens = [len(t) for t in texts]
    print(np.average(text_lens))
    print(np.min(text_lens))
    print(np.max(text_lens))
    words = [w for t in texts for w in t]

    print("Most common words")
    wc = most_common_words(words)
    print(len(wc))
    for word, occ in wc.most_common(100):
        print(word)

    top_100 = wc.most_common(100)
    print(top_100)

    selected = Counter(token for token in wc.elements() if wc[token] > 5)
    #print(len(selected))

    top_1000 = wc.most_common(1000)

    plt.bar([t[0] for t in top_100], [t[1] for t in top_100])
    plt.savefig("Most_common_words.png")

    exit(99)
