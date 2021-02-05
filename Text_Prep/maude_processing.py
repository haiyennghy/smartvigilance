import sys
sys.path.append("../smartvigilance/")

import string
import stanza
from nltk.corpus import stopwords
import re
import os

from DataImporter.MAUDE.maude_dset import Maude_pd_dataset
from DataImporter.utils import load_pkl
from Text_Prep.valid_tokens import ValidTokens



class MaudePreprocessor():

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        stanza.download('en', package='genia', processors='tokenize,pos,lemma', logging_level='WARN')
        self.nlp = stanza.Pipeline(lang='en', package='genia', processors='tokenize', use_gpu=self.use_gpu, logging_level='WARN')
        self.pretokenized_lemmatizer = stanza.Pipeline(lang='en', package='genia', processors='tokenize,pos,lemma',
                                            tokenize_pretokenized=True, use_gpu=self.use_gpu, logging_level='WARN')
        self.additional_stop_words = ["'s"]
        self.stopwords = stopwords.words('english') + self.additional_stop_words
        self.valid_tokens = ValidTokens().tokens


    def split_units(self, sent_str):
        units = ValidTokens().units
        for unit in units:
            # could be float number !
            regex = r'[0-9]+'+re.escape(unit)+r'\s'
            matches = re.findall(regex, sent_str)
            for match in matches:
                match_phrase = match.split(unit)[0]
                sent_str = sent_str.replace(match, match_phrase + ' ' + unit + ' ')
        return sent_str


    # Shorten all repetitions of 3 or more (except digits)
    def shorten_char_repetitions(self, sent_str):
        punct = string.punctuation
        regex = r'([^0-9])\1{3,}'
        sent_str = re.sub(regex, r'\1', sent_str)
        return sent_str


    def split_tokens(self, sent_str):
        for tok in sent_str.split():
            # trailing punctuation split off
            if tok[-1] in ['.',',',':', '"'] and len(tok) > 1:
                new_tok = tok[:-1]+' '+tok[-1]
                sent_str = sent_str.replace(tok, new_tok)
                tok = new_tok
            # punctuation at beginning of token split
            if tok[0] in ['"','.', ':'] and len(tok) > 1:
                new_tok = tok[0]+' '+tok[1:]
                sent_str = sent_str.replace(tok, new_tok)
                tok = new_tok
            # split alternatives with slash
            if "/" in tok and tok not in self.valid_tokens and len(tok) > 1:
                split_tok = tok.split('/')
                new_tok = " / ".join(split_tok)
                sent_str = sent_str.replace(tok, new_tok)
        return sent_str


    # makes all lowercase and replaces spanish question mark
    def pre_tokenizer_clean(self, text):
        text = text.lower()
        text = self.shorten_char_repetitions(text)
        whitespace_tok = text.split()
        tok_list = []
        for tok in whitespace_tok:
            match = re.search(r'[A-Za-z]+¿s', tok)
            if match:
                tok = tok.replace('¿', "'")
            else:
                tok = tok.replace('¿', '')
            tok_list.append(tok)
        text = " ".join(tok_list)
        return text


    def stanza_tokenizer(self, text):
        doc = self.nlp(text)
        tokenized = []
        for sent in doc.sentences:
            tokenized += [token.text for token in sent.words]
        return tokenized


    # our tokenizer, from raw input to finish, only tokenization (no token removal or lemmatization)
    def tokenizer(self, text):
        text = self.pre_tokenizer_clean(text)
        tokenized = self.stanza_tokenizer(text)
        output = self.manual_token_corrections(tokenized)
        return output


    def lemmatizer(self, tokenized_text):
        doc = self.pretokenized_lemmatizer([tokenized_text])
        lemma = []
        for sent in doc.sentences:
            lemma += [token.lemma for token in sent.words]
        return lemma


    def clean(self, tokenized_text):
        clean_text = []
        for tok in tokenized_text:
            if tok not in self.stopwords and tok not in string.punctuation and len(tok) > 1:
                clean_text.append(tok)
        return clean_text


    # merge or split valid tokens that got mis-tokenized
    def manual_token_corrections(self, seg_text):
        sent_str = " ".join(seg_text)
        for token in self.valid_tokens:
            matches = re.findall(self.valid_tokens[token], sent_str)
            for match in matches:
                match_phrase = match.replace(' ','')
                sent_str = sent_str.replace(match, ' '+match_phrase+' ')
        sent_str = self.split_units(sent_str)
        sent_str = self.split_tokens(sent_str)
        seg_text = sent_str.split(' ')
        seg_text = [token for token in seg_text if token != '']
        return seg_text


    # complete pipeline: raw input, tokenize, lemmatize and clean, output: list of tokens
    def pipe(self, text, only_tokens=False):
        output = self.tokenizer(text)
        if not only_tokens:
            output = self.lemmatizer(output)
        output = self.clean(output)
        return output



if __name__ == "__main__":
    P = MaudePreprocessor()
    pkl_name = "100000_random_entries_prod_codes.pkl"
    pkl = load_pkl(os.path.join("data", "MAUDE", pkl_name))

    savepath = os.path.join("data", "tokenized", pkl_name)
    pkl = load_pkl(os.path.join("data", "MAUDE", pkl_name))
    subset_2 = Maude_pd_dataset(pkl)
    pd_texts = subset_2.get_all_report_texts()
    #pd_texts = subset_2.get_all_report_texts(mode="reports+texts")
    for i, (idx, row) in enumerate(pd_texts.iterrows()):
        try:
            #prep = P.pipe(row["mdr_text"]["text"])
            prep = P.pipe(row["text"])
            pd_texts.at[idx, "tokenized text"] = prep
        except:
            pass
