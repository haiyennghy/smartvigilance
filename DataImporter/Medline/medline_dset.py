from torch.utils.data import Dataset



class Medline_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["PubmedArticleSet"]["PubmedArticle"])

    def get_entry(self, id):
        return self.dataset[str(id)]
