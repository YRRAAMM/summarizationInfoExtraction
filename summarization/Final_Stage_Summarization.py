import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

nltk.download('stopwords')
nltk.download('punkt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def bertSent_embedding(sentences):
    marked_sent = ["[CLS] " + item + " [SEP]" for item in sentences]
    tokenized_sent = [tokenizer.tokenize(item) for item in marked_sent]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(item) for item in tokenized_sent]
    tokens_tensor = [torch.tensor([item]) for item in indexed_tokens]
    segments_ids = [[1] * len(item) for ind, item in enumerate(tokenized_sent)]
    segments_tensors = [torch.tensor([item]) for item in segments_ids]

    bert_model.eval()

    encoded_layers_list = []
    for i in range(len(tokens_tensor)):
        with torch.no_grad():
            encoded_layers, _ = bert_model(tokens_tensor[i], segments_tensors[i])
        encoded_layers_list.append(encoded_layers)

    token_vecs_list = [layers[11][0] for layers in encoded_layers_list]
    sentence_embedding_list = [torch.mean(vec, dim=0).numpy() for vec in token_vecs_list]

    return sentence_embedding_list

def kmeans_sumIndex(sentence_embedding_list):
    try:
        n_clusters = 5
        kmeans = KMeans(n_clusters=int(n_clusters))
        kmeans = kmeans.fit(sentence_embedding_list)
        sum_index, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embedding_list, metric='euclidean')
        sum_index = sorted(sum_index)
        return sum_index
    except:
        n_clusters = 1
        kmeans = KMeans(n_clusters=int(n_clusters))
        kmeans = kmeans.fit(sentence_embedding_list)
        sum_index, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embedding_list, metric='euclidean')
        sum_index = sorted(sum_index)
        return sum_index

def bertSummarize(text):
    sentences = sent_tokenize(text)
    sentence_embedding_list = bertSent_embedding(sentences)
    sum_index = kmeans_sumIndex(sentence_embedding_list)
    summary = ' '.join([sentences[ind] for ind in sum_index])
    return summary

t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

def summarize_text(text):
    ext_summary = bertSummarize(text)
    inputs = t5_tokenizer.encode("summarize:" + ext_summary, return_tensors="pt", max_length=512, padding='max_length', truncation=True, add_special_tokens=True)
    summary_ids = t5_model.generate(inputs, num_beams=2, no_repeat_ngram_size=3, length_penalty=2.0, min_length=100, max_length=200)
    output = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output
