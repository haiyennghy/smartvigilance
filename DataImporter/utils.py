import json
import pickle
import xmltodict
import os
import pandas as pd

def save_dict_as_pkl(dict, name):
    if name[-4:] != ".pkl":
        name = name + ".pkl"

    with open(name, 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

def save_pd_in_chunks(pd_data, folder_path, rows_per_file):
    """
    Split a big pandas dataframe into subframes of a certain size and save each as a separate file in one folder.

    :param pd_data:
    :param folder_path:
    :param rows_per_file:
    :return:
    """

    pd_data = pd_data.reset_index(drop=True)
    print(pd_data)

    num_subframes = int(len(pd_data)/rows_per_file)
    print(num_subframes)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    for i in range(num_subframes):
        subframe = pd_data[i*rows_per_file:(i+1)*rows_per_file]
        print(subframe)

        subframe.to_pickle(os.path.join(folder_path, str(i*rows_per_file) + "-" + str((i+1)*rows_per_file)))

    """Remaining rows"""
    endframe = pd_data[num_subframes*rows_per_file:]
    print(endframe)

    endframe.to_pickle(os.path.join(folder_path, str(num_subframes*rows_per_file) + "-end"))


def read_all_pd_chunks(folder_path):
    """
    Read all chuncks, i.e. pd sub-dataframes and combine them to a single pandas dataframe

    :param folder_path:
    :return:
    """
    pd_data = pd.DataFrame()

    for file in os.listdir(folder_path):
        print("Reading", file)

        with open(os.path.join(folder_path, file), 'rb') as f:
            pd_subframe = pickle.load(f)

            pd_data = pd_data.append(pd_subframe)


    pd_data = pd_data.reset_index(drop=True)
    return pd_data


def load_pkl(path):

    if path[-4:] != ".pkl":
        path = path + ".pkl"

    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

def open_xml_as_dict(path):
    with open(path + '.xml') as f:
        return xmltodict.parse(f.read())


def open_json(path):
    with open(path) as json_file:
        return json.load(json_file)
