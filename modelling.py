from DataImporter.utils import open_json, load_pkl
import os
from DataImporter.MAUDE.maude_dset import Maude_pd_dataset

import pandas as pd

pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 100000)
pd.set_option('display.max_rows', 50)

dset = open_json(os.path.join("data", "MAUDE", "2020q1", "device-event-0001-of-0004.json"))
m = Maude_pd_dataset(dset)
print(m.dataset)
print(m.get_all_report_texts())

pkl = load_pkl(os.path.join("data", "MAUDE", "100000_random_entries_prod_codes.pkl"))
subset_2 = Maude_pd_dataset(pkl)
print(subset_2.get_all_report_texts())

exit(99)

"""
print(m[0])
print(m.get_report_number(1))
print(m.get_product_problems(0))

list = list()
for i in range(len(m)):
    pp = m.get_product_problems(i)
    for p in pp:
        list.append(p)

for c in Counter(list).most_common():
    print(c)

exit(99)
"""

for entry in dset["results"]:
    if len(entry["device"]) > 1:
        print(entry["mdr_report_key"])
        for d in entry["device"]:
            print(d)
        #print(entry["device"])
        try:
            print(entry["product_problems"])
        except:
            print("No Prod Problem")

        print("\n\n")
