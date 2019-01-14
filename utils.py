# coding: utf-8

# In[1]:

import xml.etree.ElementTree
from stanfordcorenlp import StanfordCoreNLP
import string
import re
import itertools
import csv
import json
from os import path
import numpy as np


# In[2]:

# POS Tags binary vectors
POS_VECTORS = {'NN': np.array([1., 0., 0., 0., 0., 0.]),
              'NNS': np.array([1., 0., 0., 0., 0., 0.]),
              'NNP': np.array([1., 0., 0., 0., 0., 0.]),
              'NNPS': np.array([1., 0., 0., 0., 0., 0.]),
              'VB': np.array([0., 1., 0., 0., 0., 0.]),
              'VBD': np.array([0., 1., 0., 0., 0., 0.]),
              'VBG': np.array([0., 1., 0., 0., 0., 0.]),
              'VBN': np.array([0., 1., 0., 0., 0., 0.]),
              'VBP': np.array([0., 1., 0., 0., 0., 0.]),
              'VBZ': np.array([0., 1., 0., 0., 0., 0.]),
              'JJ': np.array([0., 0., 1., 0., 0., 0.]),
              'JJR': np.array([0., 0., 1., 0., 0., 0.]),
              'JJS': np.array([0., 0., 1., 0., 0., 0.]),
              'RB': np.array([0., 0., 0., 1., 0., 0.]),
              'RBR': np.array([0., 0., 0., 1., 0., 0.]),
              'RBS': np.array([0., 0., 0., 1., 0., 0.]),
              'IN': np.array([0., 0., 0., 0., 1., 0.]),
              'CC': np.array([0., 0., 0., 0., 0., 1.])
             }

# Stanford core nlp list of tags
POS_TAGS = { '\'\'': 0, ',': 1, '.': 2, ':': 3, '``': 4, 'CC': 5, 'CD': 6, 'DT': 7, 'EX': 8, 'FW': 9, 'IN': 10, 'JJ': 11, 'JJR': 12, 'JJS': 13, 'LS': 14, 'MD': 15, 'NN': 16, 'NNP': 17, 'NNPS': 18, 'NNS': 19, 'PDT': 20, 'POS': 21, 'PRP': 22, 'PRP$': 23, 'RB': 24, 'RBR': 25, 'RBS': 26, 'RP': 27, 'SYM': 28, 'TO': 29, 'UH': 30, 'VB': 31, 'VBD': 32, 'VBG': 33, 'VBN': 34, 'VBP': 35, 'VBZ': 36, 'WDT': 37, 'WP': 38, 'WP$': 39, 'WRB': 40 } 


# In[3]:

# Load sentic2vec
def load_sentic2vec_model():
    print("Loading sentic2vec Model")
    sentic2vec = {}
    with open('sentic2vec.csv', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            head, *tail = row
            tail = np.array([float(i) for i in tail])
            sentic2vec.update({head: tail})
    print("Done.", len(sentic2vec), " words loaded!")
    return sentic2vec

def load_glove_model():
    glove_file_path = "glove.6B/glove.6B.300d.txt"
    print("Loading Glove Model")
    f = open(glove_file_path,'r', encoding="utf8")
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


# In[4]:

def build_dict(aspects):
    myDict = {}
    for aspect in aspects:
        begin = int(aspect["from"])
        terms = aspect["term"].split(" ")
        position = begin + len(terms[0])
        myDict[begin] = 1
        for term in terms[1:]:
            position += 1
            myDict[position] = 2
            position = position + len(term)
    return myDict

def get_pos_vector(text_pos_tags, word, word_before_clean):
    original_word = __get_original_word(word)

    for i in range(len(text_pos_tags)):
        token = text_pos_tags[i][0]
        token_tag = text_pos_tags[i][1]
        
        if token.lower() == word or token.lower() == word_before_clean.lower() or token.lower() in original_word:
            del text_pos_tags[i]
            break
    
    pos_vector = POS_VECTORS.get(token_tag, np.array([0., 0., 0., 0., 0., 0.])) # get POS binary vector
    
    return text_pos_tags, pos_vector
    

def __remove_punctuation(word):
    return word.translate(str.maketrans('','',string.punctuation))

def __get_original_word(word):
    if word == "is": return "'s"
    elif word == "have": return "'ve"
    elif word == "shall": return "sha"
    elif word == "will": return ["wo", "'ll"]
    elif word == "can": return "ca"
    elif word == "not": return "n't"
    elif word == "am": return "'m"
    elif word == "are": return "'re"
    elif word == "would": return "'d"
    else: return word
    
def clean_text(word, label):
    word = re.sub(r"\'s", " is", word)
    word = re.sub(r"\'ve", " have", word)
    word = re.sub(r"shan't", "shall not", word)
    word = re.sub(r"won't", "will not", word)
    word = re.sub(r"can't", "can not", word)
    word = re.sub(r"n't", " not", word)
    word = re.sub(r"i'm", "i am", word)
    word = re.sub(r"\'re", " are", word)
    word = re.sub(r"\'d", " would", word)
    word = re.sub(r"\'ll", " will", word)
    word = __remove_punctuation(word)
    word = word.lower()
    return [(w, label) for w in word.split(" ")]



# In[5]:

def create_inputs(words_n_labels_n_pos_vectors, embedding_model, use_pos_vectors):
    train_lstm = []
    train_5_gram = []
    labels = []
    embedding_total_size = 300
    total_words = len(words_n_labels_n_pos_vectors)
    total_zeros = 0

    if (use_pos_vectors == True):
        embedded_words = [np.concatenate((embedding_model.get(word, np.zeros(300)), pos_vector)) for (word, label, pos_vector) in words_n_labels_n_pos_vectors]
        total_zeros = sum([0 if word in embedding_model else 1 for (word, label, pos_vector) in words_n_labels_n_pos_vectors])
        embedding_total_size = 306
    else:
        words_n_labels = [(t[0], t[1]) for t in words_n_labels_n_pos_vectors]
        embedded_words = [embedding_model.get(word, np.zeros(300)) for (word, label) in words_n_labels]
        total_zeros = sum([0 if word in embedding_model else 1 for (word, label) in words_n_labels])
        

    for i in range(len(words_n_labels_n_pos_vectors)):
        label = words_n_labels_n_pos_vectors[i][1]
        
        # create one embedding matrix for each example: [pre-word1, pre-word2, main-word, pos-word1, pos-word2]
        embedding_matrix = np.zeros((5, embedding_total_size))

        embedding_matrix[0] = embedded_words[i-2] if i - 2 >= 0 else np.zeros(embedding_total_size) # pre-word1
        embedding_matrix[1] = embedded_words[i-1] if i - 1 >= 0 else np.zeros(embedding_total_size) # pre-word2
        embedding_matrix[2] = embedded_words[i] # main-word
        embedding_matrix[3] = embedded_words[i+1] if i + 1 < len(words_n_labels_n_pos_vectors) else np.zeros(embedding_total_size) # pos-word1
        embedding_matrix[4] = embedded_words[i+2] if i + 2 < len(words_n_labels_n_pos_vectors) else np.zeros(embedding_total_size) # pos-word2

        # 5-gram input format
        train_5_gram.append(embedding_matrix)

        # lstm input format
        train_lstm.append(embedded_words[i]) # main-word

        labels.append(label)
        
    return train_lstm, train_5_gram, labels, total_words, total_zeros


# In[22]:

def read_xml_file(file_path, embedding_type='glove', use_pos_vectors=False):
    TEST_IDS = []
    if (file_path == 'data/Laptops_Train.xml'):
        TEST_IDS = [2128,81,89,353,347,1813,655,1615,1670,2443,764,177,3012,1479,2937,439,2925,929,2077,225,2756,2863,1393,1837,921,2817,1896,341,36,990,2275,2908,214,2777,467,47,361,168,2642,1000,2135,1733,1822,3064,1646,2988,2828,121,1967,1776,2626,2967,1800,2993,1037,1270,1834,2451,605,1900,2871,844,1634,200,416,68,2746,282,1532,2238,692,78,2245,415,2710,1671,2703,66,285,1609,2661,1577,1287,2523,2785,832,2037,1429,2130,1446,2259,2695,1506,3047,1042,2151,1732,1063,271,1674]
    elif (file_path == 'data/Restaurants_Train.xml'):
        TEST_IDS = [813,1579,2707,3126,2882,1609,3018,3292,2041,2609,1194,3440,870,2507,586,3081,1884,2077,201,3049,1242,2211,3343,2141,83,381,3315,1709,193,2459,1293,2105,2152,3358,2751,882,1124,3546,3020,746,3170,2171,2811,762,339,217,3478,1180,3372,776,3208,2673,1380,3641,2806,2630,924,3411,2164,3077,745,1273,179,2460,3045,1698,109,1057,2186,3471,1401,1112,3368,3495,2451,948,2202,2740,537,1548,134,2708,1455,2591,1593,1769,3575,1105,2575,742,1471,1656,2769,3681,2205,1159,628,2912,3188,3041]

    NLP = StanfordCoreNLP(path.relpath(r'./stanford-corenlp-full-2018-10-05/'))
    
    embedding_model = {}
    if (embedding_type == 'glove'):
        embedding_model = load_glove_model()
    elif (embedding_type == 'sentic'):
        embedding_model = load_sentic2vec_model()

    train_list_5_gram = []
    train_list_neural_net = []
    labels_list_neural_net = []
    train_list_lstm = []
    labels_list_lstm = []
    sentences_length_lstm = []
    original_sentences = []
    total_words = 0
    total_zeros = 0

    # parse
    e = xml.etree.ElementTree.parse(file_path).getroot()
    for sentence in e.iter('sentence'):
        id_ = sentence.get('id')
        
        # remove test data from train data
        if int(id_) in TEST_IDS and (file_path == 'data/Laptops_Train.xml' 
                                     or file_path == 'data/Restaurants_Train.xml'): continue
        
        text = sentence.find('text').text
        aspects = [{"term": a.get('term'), 
                    "polarity": a.get('polarity'), 
                    "from": a.get('from'), 
                    "to": a.get('to')} for a in sentence.iter('aspectTerm')]

        words_n_labels_n_pos_vectors = []
        aspectsDict = build_dict(aspects)

        text_pos_tags = NLP.pos_tag(text) # get POSTags using stanford core NLP

        i = 0
        while i < len(text):
            if i in aspectsDict:
                label = aspectsDict[i]
            else:
                label = 0

            word = ''
            while i < len(text) and text[i] != ' ':
                word += text[i]
                i += 1
                        
            clean_words_n_labels = clean_text(word, label)

            # get POS vectors
            for (clean_word, label) in clean_words_n_labels:
                text_pos_tags, pos_vector = get_pos_vector(text_pos_tags, clean_word, word) # update 'text_pos_tags' and get 'pos_vector'
                word_label_pos_vector = (clean_word, label, pos_vector)
                
                words_n_labels_n_pos_vectors.append(word_label_pos_vector)

            i += 1

        train_lstm, train_5_gram, labels, num_words, num_zeros = create_inputs(words_n_labels_n_pos_vectors, embedding_model, use_pos_vectors)
        
        train_list_neural_net += train_5_gram
        labels_list_neural_net += labels
        train_list_5_gram.append(train_5_gram)
        train_list_lstm.append(train_lstm)
        labels_list_lstm.append(labels)
        
        total_words += num_words
        total_zeros += num_zeros
        
        sentences_length_lstm.append(len(words_n_labels_n_pos_vectors))
        original_sentences.append([word for (word, label, pos_vector) in words_n_labels_n_pos_vectors])


    print("Total words in the set:", total_words)
    print("Total words not recognized by the embedding:", total_zeros)

    NLP.close()
    del embedding_model
    
    return train_list_neural_net, labels_list_neural_net, train_list_lstm, labels_list_lstm, train_list_5_gram, sentences_length_lstm, original_sentences


# In[6]:

def read_pos_tagging_file(file_path):
    train_list_5_gram = []
    train_list_lstm = []
    labels_list_lstm = []
    sentences_length_lstm = []
    original_sentences = []
    total_words = 0
    total_zeros = 0
    
    embedding_model = load_glove_model()
    
    f = open(file_path,'r')
    for line in f:
        token_tags = json.loads(line)['tags']
        words_labels = [(tk_tg['tk'].lower(), POS_TAGS[tk_tg['tg']]) for tk_tg in token_tags]

        train_lstm, train_5_gram, labels, num_words, num_zeros = create_inputs(words_labels, embedding_model, False)

        train_list_5_gram.append(train_5_gram)
        train_list_lstm.append(train_lstm)
        labels_list_lstm.append(labels)
        
        total_words += num_words
        total_zeros += num_zeros
        
        sentences_length_lstm.append(len(words_labels))
    
    print("Total words in the set:", total_words)
    print("Total words not recognized by the embedding:", total_zeros)
    
    del embedding_model
    
    return train_list_lstm, labels_list_lstm, train_list_5_gram, sentences_length_lstm

# In[ ]:



