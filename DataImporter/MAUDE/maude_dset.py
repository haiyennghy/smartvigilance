from torch.utils.data import Dataset
import pandas as pd
import warnings



class Maude_dataset(Dataset):
    def __init__(self, json):
        self.dataset = json["results"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_entry(self, id):
        return self.dataset[id]

    def get_report_number(self, id):
        return self.get_entry(id)["report_number"]

    def get_texts(self, id):
        return self.get_entry(id)["mdr_text"]

    def get_devices(self, id):
        return self.get_entry(id)["device"]

    def get_device_report_product_code(self, id):

        device_report_product_codes = []
        for device in self.get_devices(id):
            device_report_product_codes.append(device["device_report_product_code"])

        return device_report_product_codes

    def get_event_type(self, id):
        return self.get_entry(id)["event_type"]

    def get_patient(self, id):
        return self.get_entry(id)["patient"]

    def get_product_problems(self, id):
        try:
            return self.get_entry(id)["product_problems"]
        except:
            return [None]

    def is_product_problem(self, id):
        return self.get_entry(id)["product_problem_flag"] == "Y"

    def get_manufacturer_name(self, id):
        return self.get_entry(id)["manufacturer_g1_name"]

    def get_date_of_event(self, id):
        return self.get_entry(id)["date_of_event"]

    def get_date_received(self, id):
        return self.get_entry(id)["date_received"]

    def get_date_report(self, id):
        return self.get_entry(id)["date_report"]

    def get_type_of_report(self, id):
        return self.get_entry(id)["type_of_report"]



















class Maude_pd_dataset(Dataset):
    """
    Class representing the MAUDE data, i.e. a pickle files created from the JSONs.
    Internally stored as pandas Dataframe
    """

    def __init__(self, data):
        try:
            self.dataset = pd.DataFrame(data["results"])
        except:
            self.dataset = pd.DataFrame(data)

            """In case the dset is transposed, i.e. columns and rows are changed transpose it"""
            if len(self.dataset.columns) > len(self.dataset):
                self.dataset = self.dataset.transpose().reset_index(drop=True)
            else:
                self.dataset = self.dataset.reset_index(drop=True)

            print("Memory:", self.dataset.memory_usage().sum() / 1024 ** 2)

            self.dataset = self.drop_unnecessary_columns()
            self.dataset = self.type_columns()

            print("Memory:", self.dataset.memory_usage().sum() / 1024 ** 2)

        #print(self.dataset)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.loc[idx]

    def get_devices(self):
        return pd.DataFrame(self.dataset["device"])

    def get_all_report_texts(self):
        """Returns all report texts, i.e. not the whole dict "mdr_text" but only the text inside mdr_text of each entry"""

        mdr_texts = pd.DataFrame.from_dict(self.dataset["mdr_text"].explode("mdr_text").to_dict(), orient="columns").transpose()
        return pd.DataFrame(mdr_texts)

    def explode(self):

        """Explode columns mdr_text and device, i.e. copy the row n times for n elements in mdr_text or device"""
        self.dataset = self.dataset.explode(column="mdr_text")
        self.dataset = self.dataset.explode("device").reset_index(drop=True)

        print("Memory:", self.dataset.memory_usage().sum() / 1024 ** 2)

    def unpack_device_column(self):
        """For each key in the device dict, add a column"""
        warnings.warn("Method can cause issues, do NOT call before tokenization")

        device = self.dataset["device"].fillna({i: {} for i in self.dataset.index})
        device = pd.json_normalize(device)

        device = device.drop(["manufacturer_d_address_2",
                              "manufacturer_d_zip_code",
                              "manufacturer_d_postal_code",
                              "manufacturer_d_zip_code_ext",
                              #"manufacturer_contact_pcountry",
                              "other_id_number",
                              #"removal_correction_number",
                              #"manufacturer_link_flag",
                              #"manufacturer_contact_extension,"
                              "openfda.fei_number"
                              ],
                             axis=1)

        device_types =          {"device_operator": "category",
                                "device_evaluated_by_manufacturer": "category",
                                "device_sequence_number": "category",
                                "manufacturer_d_state": "category",
                                "implant_flag": "bool",
                                "device_availability": "category",
                                "date_removed_flag": "category",
                                "device_report_product_code": "category",
                                "openfda.regulation_number": "category",
                                "openfda.medical_specialty_description": "category",
                                "openfda.device_class": "category",
                                "openfda.device_name": "category",
                                "baseline_510_k__flag": "category",
                                "baseline_510_k__exempt_flag": "category",
                                "baseline_510_k__number": "category",
                                }

        for col, type in device_types.items():
            device = device.astype({col: type})

        self.dataset = pd.concat([self.dataset.drop(["device"], axis=1), device], axis=1)

        print("Memory:", self.dataset.memory_usage().sum() / 1024 ** 2)

    def unpack_mdr_text_column(self):
        """For each key in mdr_text dict, add a column"""

        mdr_text = self.dataset["mdr_text"].fillna({i: {} for i in self.dataset.index})
        #print(mdr_text.isnull().any())
        mdr_text = pd.json_normalize(mdr_text)
        #self.dataset["mdr_text"].apply(pd.Series) #very slow
        mdr_text = mdr_text.drop(["patient_sequence_number"], axis=1)
        mdr_text = mdr_text.astype({"text_type_code": "category"})
        self.dataset = pd.concat([self.dataset.drop(["mdr_text"], axis=1), mdr_text], axis=1)

        print("Memory:", self.dataset.memory_usage().sum() / 1024 ** 2)

    """
    def unpack_openfda_column(self):
        self.dataset = pd.concat([self.dataset.drop(["openfda"], axis=1),
                                  self.dataset["openfda"].apply(pd.Series)], axis=1)

        self.dataset = self.drop_unnecessary_columns()
        self.dataset = self.type_columns()

        print("Memory:", self.dataset.memory_usage().sum() / 1024 ** 2)
    """

    def unpack_patient_columns(self):
        raise NotImplementedError()

    def type_columns(self):
        """Change the columns dtype to more memory efficient ones"""

        column_types = {"mdr_report_key": "int32",
                        #"manufacturer_contact_plocal": "int32",
                        #"manufacturer_contact_phone_number": "int32",
                        #"date_of_event": "int32",
                        "date_received": "int32",
                        #"manufacturer_g1_zip_code": "int32",
                        #"manufacturer_contact_pcity": "int32",
                        #"date_report": "int32",
                        #"date_changed": "int32",
                        #"date_added": "int32",
                        #"manufacturer_contact_exchange": "int32",
                        #"date_manufacturer_received": "int32",
                        #"manufacturer_contact_zip_code": "int32",
                        #"date_report_to_fda": "int32",

                        "single_use_flag": "bool",
                        "reprocessed_and_reused_flag": "bool",
                        "health_professional": "bool",
                        "manufacturer_link_flag": "bool",
                        "adverse_event_flag": "bool",

                        "previous_use_code": "category",
                        "initial_report_to_fda": "category",
                        "event_type": "category",
                        "manufacturer_contact_country": "category",
                        "manufacturer_g1_country": "category",
                        "manufacturer_g1_state": "category",
                        "manufacturer_contact_state": "category",
                        #"manufacturer_country": "category",
                        #"manufacturer_name": "category",
                        #"manufacturer_zip_code_ext": "category",
                        #"manufacturer_zip_code": "category",
                        #"manufacturer_address_1": "category",
                        #"manufacturer_address_2": "category",
                        #"manufacturer_state": "category",
                        #"report_to_fda": "category",
                        #"report_to_manufacturer": "category",
                        #"distributor_zip_code": "category",
                        #"distributor_zip_code_ext": "category",
                        #"distributor_address_2": "category",
                        #"distributor_address_1": "category",
                        #"distributor_city": "category",
                        #"distributor_name": "category",
                        #"distributor_state": "category",
                        "report_source_code": "category",
                        #"number_devices_in_event": "category",
                        #"event_key": "category",
                        "product_problem_flag": "category",
                        "event_location": "category",
                        "remedial_action": "category",
                        "type_of_report": "category",
                        "reporter_occupation_code": "category",
                        "report_to_fda": "category",
                        "manufacturer_d_country": "category",
                        "device_operator": "category",
                        "device_evaluated_by_manufacturer": "category",
                        "device_sequence_number": "category",
                        "manufacturer_d_state": "category",
                        "implant_flag": "category",
                        "device_availability": "category",
                        "baseline_510_k__exempt_flag": "category",
                        "baseline_510_k__flag": "category",
                        "baseline_510_k__number": "category",
                        }

        for col, dtype in column_types.items():
            try:
                self.dataset = self.dataset.astype({col: dtype})
            except:
                continue
        """
        for attr in self.dataset.columns:
            if attr == "mdr_text":
                continue
            if attr == "device":
                continue
            print(self.dataset[attr].value_counts())
        """

        return self.dataset

    def drop_unnecessary_columns(self):
        """Drop columns that hold values, that are not informative, e.g. due to mostly empty fields"""

        uninformative_columns = ["manufacturer_contact_zip_ext",  #mostly empty
                                 "manufacturer_g1_address_2",  #mostly empty
                                 "manufacturer_g1_zip_code_ext",  # mostly empty
                                 "manufacturer_contact_t_name",  #mostly empty, otherwise Mr, Mrs, ...
                                 "manufacturer_contact_address_2",  #mostly empty
                                 "manufacturer_address_2",  #always empty
                                 "manufacturer_address_1",  #always empty
                                 #"exemption_number",               #always empty
                                 "distributor_zip_code_ext",  #always empty
                                 "manufacturer_zip_code",  #always empty
                                 "manufacturer_city",  #always empty
                                 "distributor_city",  #always empty
                                 "distributor_state",  #always empty
                                 "event_key",  #always empty
                                 "number_devices_in_event",  #always empty
                                 "manufacturer_name",  #always empty
                                 "report_to_manufacturer",  #always empty
                                 "manufacturer_zip_code_ext",  #always empty
                                 "distributor_address_1",  #always empty
                                 "manufacturer_state",  #always empty
                                 "distributor_address_2",  #always empty
                                 "manufacturer_postal_code",  #always empty
                                 "manufacturer_country",  #always empty
                                 "number_patients_in_event",  #always empty
                                 "distributor_name",  #always empty
                                 "distributor_zip_code",  #always empty
                                 "date_removed_flag",  #mostly empty
                                 "manufacturer_d_address_2",  #unneccesarry
                                 "manufacturer_d_zip_code",  #unneccessary
                                 "manufacturer_d_postal_code",  #unneccessary
                                 "manufacturer_d_zip_code_ext",  # unneccessary
                                 "patient_sequence_number",  #always 1
                                 "manufacturer_contact_pcountry",  # unneccessary
                                 "removal_correction_number",  # mostly empty
                                 "manufacturer_link_flag",  #always True
                                 "manufacturer_contact_extension",  # unneccessary
                                 ]


        for col in uninformative_columns:
            try:
                self.dataset = self.dataset.drop(col, axis=1)
            except:
                continue

        return self.dataset

