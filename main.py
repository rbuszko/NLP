import pandas as pd
import re
import string
import nltk
import collections
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import random
import numpy as np
from sklearn.decomposition import PCA
import gensim, logging

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(' +', ' ', text)
    return text

def spacy_tokenizer(document):
    tokens = nlp(document)
    tokens = [token.lemma_ for token in tokens if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.lemma_.strip()!= '')]
    return tokens

def stemm_text(tokens):
    stemmer = nltk.SnowballStemmer("english")
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def dummy(doc):
    return doc

def word2vec(tokens):
    if len(tokens) > 0:
        word2vec = model.wv[tokens]
        return np.average(word2vec, axis=0)
    else:
        # When there is no token in tokens
        return np.zeros(100)

if __name__ == '__main__':
    # To get results faster
    spacy.prefer_gpu()

    # Code for 1 database
    # df = pd.read_csv("data/emails.csv", encoding='latin-1')
    # df.columns = ['text', 'label']
    # df['label'] = df['label'].replace([1],'spam')
    # df['label'] = df['label'].replace([0],'ham')

    # Code for 2 database
    # df = pd.read_csv("data/spam_ham_dataset.csv", encoding='latin-1')
    # df.drop(['label_num', 'Unnamed: 0'], axis=1, inplace=True)
    # df = df[['text', 'label']]

    # Code for 3 database
    df = pd.read_csv("data/spam.csv", encoding='latin-1')
    df.dropna(how="any", inplace=True, axis=1)
    df.columns = ['label', 'text']
    df = df[['text', 'label']]

    nlp = spacy.load("en_core_web_sm")
    # Calculate text
    df['text_length'] = df['text'].apply(lambda x: len(x.split(' ')))
    df['tokenized'] = df['text'].apply(spacy_tokenizer)
    df['stemmer'] = df['tokenized'].apply(stemm_text)

    ##################
    ## Bag of words ##
    ##################

    # # Bag of words
    # cv = CountVectorizer(tokenizer=dummy, preprocessor=dummy,)
    # counts = cv.fit_transform(df['stemmer'])
    # df_counts = pd.DataFrame(counts.A, columns=cv.get_feature_names_out())

    # # Need to normalize dataframe before PCA
    # df_counts_normalized = df_counts.copy()
    # for column in df_counts_normalized.columns:
    #     df_counts_normalized[column] = (df_counts_normalized[column] - df_counts_normalized[column].min()) / (df_counts_normalized[column].max() - df_counts_normalized[column].min())   

    # # Implement PCA
    # pca = PCA(n_components=2)
    # embeddings_2d = pca.fit_transform(df_counts_normalized)

    # # Connecting everything
    # final_data = pd.DataFrame(embeddings_2d, columns=['pca1', 'pca2'])
    # final_data['text_length'] = df['text_length']
    # final_data['label'] = df['label']
    
    #################
    ## Word to vec ##
    #################

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(df['stemmer'], min_count=1)
    df['word_to_vec'] = df['stemmer'].apply(word2vec)

    # Implement PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(pd.DataFrame.from_records(df['word_to_vec']))

    # Connecting everything
    final_data = pd.DataFrame(embeddings_2d, columns=['pca1', 'pca2'])
    final_data['text_length'] = df['text_length']
    final_data['label'] = df['label']
    print(final_data)
