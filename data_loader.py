# -*- coding: utf-8 -*-

import os
import nltk
import numpy as np
import scipy.io
import pandas as pd

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

import tokenization

from bert_serving.client import BertClient


def load_stackoverflow(data_path='data/stackoverflow/'):

    # load SO embedding
    with open(data_path + 'vocab_withIdx.dic', 'r') as inp_indx, \
            open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
            open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        emb_index = inp_dic.readlines()
        emb_vec = inp_vec.readlines()
        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec):
            word = index_word[index.replace('\n', '')]
            word_vectors[word] = np.array(list((map(float, vec.split()))))

        del emb_index
        del emb_vec

    with open(data_path + 'title_StackOverflow.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        # all_lines = inp_txt.readlines()
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    pca = PCA(n_components=1)
    pca.fit(all_vector_representation)
    pca = pca.components_

    XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

    XX = XX1

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    with open(data_path + 'label_StackOverflow.txt') as label_file:
        y = np.array(list((map(int, label_file.readlines()))))
        print(y.dtype)

    return XX, y


def load_search_snippet2(data_path='data/SearchSnippets/new/'):
    mat = scipy.io.loadmat(data_path + 'SearchSnippets-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'SearchSnippets_vocab2idx.dic', 'r') as inp_indx:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'SearchSnippets.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        all_lines = [line for line in all_lines]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(12340, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_

    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def load_biomedical(data_path='data/Biomedical/'):
    mat = scipy.io.loadmat(data_path + 'Biomedical-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'Biomedical_vocab2idx.dic', 'r') as inp_indx:
        # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
        # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'Biomedical.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        # print(sum([len(line.split()) for line in all_lines])/20000) #avg length
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, random_state=rand_seed, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_
    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def sip_bert_stackoverflow(data_path='data/stackoverflow/'):
    tokenizer = tokenization.FullTokenizer(
        vocab_file='/sdb/hugo/wwm_uncased_L-24_H-1024_A-16/vocab.txt', do_lower_case='True')
    bc = BertClient(port=5555, port_out=5556, check_length=False)
    with open(data_path + 'title_StackOverflow.txt', 'r') as inp_txt:
        # all_lines = inp_txt.readlines()[:-1]
        all_lines = inp_txt.readlines()
        all_lines = [line for line in all_lines]
        all_lines_tokenization = [tokenizer.tokenize(line) for line in all_lines]
        text_file = " ".join([" ".join(line) for line in all_lines_tokenization])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_lines_embedding = bc.encode(all_lines)
        all_vector_representation = np.zeros([len(all_lines), 1024])

        for i, token_sentence in enumerate(all_lines_tokenization):
            embedding_sentence = all_lines_embedding[i][1:]
            assert len(embedding_sentence) >= len(token_sentence)

            j = 0
            sent_rep = None
            for k, token in enumerate(token_sentence):
                tv = embedding_sentence[k]
                j = k
                weight = 0.1 / (0.1 + unigram[token])
                if k == 0:
                    sent_rep = tv * weight
                else:
                    sent_rep += tv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_

    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    with open(data_path + 'label_StackOverflow.txt') as label_file:
        y = np.array(list((map(int, label_file.readlines()))))

    return XX, y

def response_bert_stackoverflow(data_path='data/stackoverflow/'):
    bc = BertClient(port=5555, port_out=5556, check_length=False)
    with open(data_path + 'title_StackOverflow.txt', 'r') as inp_txt:
        # all_lines = inp_txt.readlines()[:-1]
        all_lines = inp_txt.readlines()
        XX = bc.encode(all_lines)

    with open(data_path + 'label_StackOverflow.txt') as label_file:
        y = np.array(list((map(int, label_file.readlines()))))

    return XX, y

def generate_tsv(data_path='data/stackoverflow/'):
    DATA_DIR = '/sdb/hugo/data/PublicSentiment/stackflow'
    with open(data_path + 'title_StackOverflow.txt', 'r') as inp_txt:
        # all_lines = inp_txt.readlines()[:-1]
        all_lines = inp_txt.readlines()
        all_lines = [' '.join(line.split()) for line in all_lines]

    with open(data_path + 'label_StackOverflow.txt') as label_file:
        y = np.array(list((map(int, label_file.readlines())))).astype(int) - 1

    data = np.column_stack((all_lines, y))

    total_count = len(data)
    print('Total:%s' % (total_count))
    data_pd = pd.DataFrame(data)
    data_pd = data_pd.take(np.random.permutation(total_count))

    train_pd = data_pd[: int(total_count * 0.8)]
    dev_pd = data_pd[int(total_count * 0.8):]
    # dev_pd = data_pd[int(total_count * 0.8): int(total_count * 0.9)]
    # test_pd = data_pd[int(total_count * 0.9):]

    train_pd.to_csv(DATA_DIR + '/train.tsv', sep='\t', header=None, index=False)
    dev_pd.to_csv(DATA_DIR + '/dev.tsv', sep='\t', header=None, index=False)
    dev_pd.to_csv(DATA_DIR + '/test.tsv', sep='\t', header=None, index=False)


def load_data(dataset_name):
    print('load data')
    if dataset_name == 'stackoverflow':
        return load_stackoverflow()
    elif dataset_name == 'biomedical':
        return load_biomedical()
    elif dataset_name == 'search_snippets':
        return load_search_snippet2()
    elif dataset_name == 'sip_bert_stackoverflow':
        return sip_bert_stackoverflow()
    elif dataset_name == 'response_bert_stackoverflow':
        return response_bert_stackoverflow()
    else:
        raise Exception('dataset not found...')

if __name__ == '__main__':
    load_data('response_bert_stackoverflow')
    # generate_tsv()