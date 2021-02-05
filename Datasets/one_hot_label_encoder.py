import pandas as pd

class OneHotEncode():
    def one_hot_encoder(self, dset, onehot_dataset_path):

        # One hot encoder
        one_hot_df = pd.get_dummies(dset["device_report_product_code"])

        # Remove labels column from original dataframe
        df = dset.drop('device_report_product_code',axis = 1)
        
        # Join the one hot encoded df with the original dataframe along columns
        df = df.join(one_hot_df)
        df.to_csv(onehot_dataset_path, sep=',', index=False)
        del(one_hot_df)

        df['list'] = df[df.columns[1:]].values.tolist()
        new_df = df[['Text', 'list']].copy()
        del(df)

        print(new_df.head())
        return new_df