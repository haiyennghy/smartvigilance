import sys
sys.path.append("../smartvigilance/")

import os
import codecs
import string
import hunspell
from collections import Counter

from DataImporter.MAUDE.maude_dset import Maude_pd_dataset
from DataImporter.utils import load_pkl
from Text_Prep.valid_tokens import ValidTokens



class MaudeVocabulary():

    def __init__(self, pickle_data):
        self.hobj = hunspell.HunSpell(os.path.join("..", "data", "hunspell", "en_US.dic.txt"),
                                      os.path.join("..", "data", "hunspell", "en_US.aff.txt"))
        ADDITIONAL_DICTIONARY = 'additional_vocabulary_terms.txt'
        terms = codecs.open(os.path.join("data", ADDITIONAL_DICTIONARY), 'r', 'utf-8').readlines()
        for term in terms:
            term = term.split()[0]
            self.hobj.add(term)

        pkl = load_pkl(os.path.join("..", "data", "tokenized", pickle_data))
        self.data = Maude_pd_dataset(pkl).dataset
        self.valid_tokens = ValidTokens().tokens
        self.vocabulary = Counter()
        self.OOV = Counter()
        # Tokens ordered by occurrence
        self.occ = Counter()



    def get_vocab(self):
        for i, (idx, row) in enumerate(self.data.iterrows()):
            try:
                tokens = row["tokenized text"]
                for token in tokens:
                    self.vocabulary[token] += 1
            except:
                pass
                #print(row)


    def OOV_words(self):
        for type in self.vocabulary:
            if not self.hobj.spell(type):
                if type not in self.valid_tokens:
                    if type not in string.punctuation:
                        self.OOV[type] = self.vocabulary[type]


    def print_numbers(self):
        print('token count: ', sum(self.vocabulary.values()))
        print('OOV tokens: ', sum(self.OOV.values()))
        print('type count: ', len(self.vocabulary))
        print('OOV types: ', len(self.OOV))
        for i in range(1,10):
            print('Tokens occuring ', i, ' times: ', self.token_occurence(i))
        OOV_hapax_count = 0
        for word in self.occ[1]:
            if word in self.OOV:
                OOV_hapax_count += 1
        print('OOV words that occur only once: ', OOV_hapax_count)
        OOV_hapax2_count = 0
        for word in self.occ[2]:
            if word in self.OOV:
                OOV_hapax2_count += 1
        print('OOV words that occur only twice: ', OOV_hapax2_count)


    def write_to_file(self, filename, oov_filename):
        vocab = open(filename, 'w')
        for type in self.vocabulary.most_common():
            vocab.write(type[0] + '\t' + str(type[1]) + '\n')
        oov = open(oov_filename, 'w')
        for type in self.OOV.most_common():
            oov.write(type[0] + '\t' + str(type[1]) + '\n')


    def tokens_ordered_by_occurence(self):
        for k, v in self.vocabulary.items():
            if self.occ[v] == 0:
                self.occ[v] = [k]
            else:
                self.occ[v].append(k)


    # How many tokens occur n times
    def token_occurence(self, n):
        return len(self.occ[n])


    def get_numbers(self):
        self.get_vocab()
        self.OOV_words()
        self.tokens_ordered_by_occurence()
        self.write_to_file('../data/vocab/100000_entries_vocab.txt', '../data/vocab/100000_entries_OOV_vocab.txt')
        self.print_numbers()


if __name__ == "__main__":
    #Vocab = MaudeVocabulary("100000_random_entries_prod_codes.pkl")
    Vocab = MaudeVocabulary("subset_2.pkl")
    Vocab.get_numbers()
