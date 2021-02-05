import pandas as pd

class CustomDatasetSplit():
    def __init__(self):
        self.train_size = 0.7
        #self.random_state = 200
        self.frac = 0.5

    def create_train_val_datasets(self, new_df, seed):
        train_dataset = new_df.sample(frac=self.train_size, random_state=seed)

        validation_dataset = new_df.drop(train_dataset.index)

        # 50% of Validation dataset and not from the entire dataset
        test_dataset = validation_dataset.sample(frac=self.frac, random_state=seed)

        # Remove samples from validation dataset who are also seen in testset
        validation_dataset = validation_dataset.drop(test_dataset.index)

        # test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)

        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)
        validation_dataset = validation_dataset.reset_index(drop=True)

        print(f"\nUnique samples in Trainset: {len(train_dataset)}, Testset: {len(test_dataset)} and Validationset: {len(validation_dataset)}")

        return train_dataset, test_dataset, validation_dataset