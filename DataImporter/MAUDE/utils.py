import os

from DataImporter.MAUDE.maude_dset import Maude_pd_dataset
from DataImporter.utils import read_all_pd_chunks, load_pkl


def read_whole_MAUDE(version):

    if version == "raw":
        pkl = read_all_pd_chunks(os.path.join("data", "MAUDE", "all_entries"))
        pd_maude_dset = Maude_pd_dataset(pkl)

    elif version == "tokenized":
        pd_maude = read_all_pd_chunks(os.path.join("data", "tokenized", "MAUDE"))
        pd_maude_dset = Maude_pd_dataset(pd_maude)

    else:
        raise ValueError("version must be one of [raw, tokenized]. Got", version)

    return pd_maude_dset


def load_maude_dset(path):

    pkl = load_pkl(path)

    pd_maude = Maude_pd_dataset(pkl)

    return pd_maude
