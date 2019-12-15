'''
This script is to prepare train/val(test) data for Charades-STA dataset
'''

import os
import json
import numpy as np
import time
import sys
#from importlib import reload
from collections import OrderedDict
from stemming.porter2 import stem

# python2
# reload(sys)
# sys.setdefaultencoding('utf-8')

unk_word = '<UNK>'
pad_word = '<PAD>'
punctuations = [':', '!', '?', '.', ';', '(', ')', '-', '_', '\n']
count_threshold = 0 # keep all words (since we are going to use Glove pre-trained word embedding vectors)

abbre_dict = {'\'s': ' is ', '\'re': ' are ', '/': ' or ', '\'m': ' am '}

## sentence preprocessing: add start word and end word, remove all punctuations, lower case
def preprocess_caption(sentence):
    # remove all punctuations
    for p in punctuations:
        sentence = sentence.replace(p,' ')
    sentence = sentence.replace(',', ' , ') # keep , notation to inforce sentence smoothness

    for abbre_word in abbre_dict.keys():
        sentence = sentence.replace(abbre_word, abbre_dict[abbre_word])
    
    words = sentence.lower().split()
    #words = [stem(word) for word in words]

    return words

## encode sentence: use id to represent a word, a sentence is encoded as a list of id
def encode_caption(sentence, vocab):
    tokens = preprocess_caption(sentence)
    tokens_id = [vocab.get(x, vocab[unk_word]) for x in tokens]
    
    return tokens_id

## build vocabulary: generate vocab .txt file and .json file 
def build_vocabulary(train_data, out_vocab_file, out_encoded_vocab_file):
    # first get word frequency
    vocab_freq = {}
    video_ids = train_data.keys()
    sentence_count = 0
    for video_id in video_ids:
        for sentence in train_data[video_id]['sentences']:
            sentence_count += 1
            tokens = preprocess_caption(sentence)
            for token in tokens:
                if token in vocab_freq.keys():
                    vocab_freq[token] = vocab_freq[token] + 1
                else:
                    vocab_freq[token] = 1
    
    # remove words with low frequency 
    print('Dictionary size: %d'%len(vocab_freq)) 
    vocabs = vocab_freq.copy()
    for word in vocab_freq.keys():
        if vocabs[word] < count_threshold:
            vocabs.pop(word)

    # add special word: 'UNK' (to represent unknown word)
    vocabs[unk_word] = 10000000 # for convenience, use a large number to make sure unk_word get id 0
    print('After removing tail words, dictionary size: %d' % len(vocabs))
    
    # sort by frequency 
    vocabs_sort = OrderedDict(sorted(vocabs.items(), key=lambda t: t[1], reverse=True))
    
    # write vocab frequency file
    vocab_freq_fid = open(out_vocab_file, 'w')
    for word in vocabs_sort.keys():
        vocab_freq_fid.write(word + ' ' + str(vocabs_sort[word]) + '\n')
    vocab_freq_fid.close()
    
    # encode 
    encoded_vocab = {}
    id = 0
    for word in vocabs_sort.keys():
        encoded_vocab[word] = id
        id = id + 1
    #encoded_vocab[pad_word] = 0
    # write encoded vocab
    print('Saving encoded dictionary ...')
    with open(out_encoded_vocab_file, 'w') as encoded_vocab_fid:
        json.dump(encoded_vocab, encoded_vocab_fid)
    
    return encoded_vocab
 

save_root = './data/'
vocabulary_file = os.path.join(save_root, 'vocabulary.txt')
encoded_vocabulary_file = os.path.join(save_root, 'word2id.json')
glove_word_embed_path = '../../download/glove.42B.300d/glove.42B.300d.txt'

# time
start_time = time.time()

print('Building train/val/test json data ...')
train_data_file = './raw_data/charades_sta_train.txt'
test_data_file = './raw_data/charades_sta_test.txt'
val_data_file = test_data_file   # since there is only train/test split, val == test

train_data = OrderedDict()
test_data = OrderedDict()
val_data = OrderedDict()

data_files = [train_data_file, val_data_file, test_data_file]
data = [train_data, val_data, test_data]

fps = 16.
for idd, data_file in enumerate(data_files):
    lines = open(data_file, 'r').readlines()
    for line in lines:
        grounding, sentence = line.strip().split('##')
        video_name, start_time, end_time = grounding.split()
        start_time, end_time = float(start_time), float(end_time)
        start_frame, end_frame = int(start_time*fps), int(end_time*fps)
        
        sentence = sentence.replace('.', '')

        if video_name not in data[idd].keys():
            data[idd][video_name] = {'timestamps': [[start_time, end_time]], \
                                     'framestamps': [[start_frame, end_frame]], \
                                    'sentences': [sentence]}
            
        else:
            data[idd][video_name]['timestamps'].append([start_time, end_time])
            data[idd][video_name]['framestamps'].append([start_frame, end_frame])
            data[idd][video_name]['sentences'].append(sentence)


train_data, val_data, test_data = data[0], data[1], data[2]
print('#Train: {}, #Val: {}, #Test: {}'.format(len(train_data.items()), len(val_data.items()), len(test_data.items())))


print('Building vocabulary ...')
encoded_vocab = build_vocabulary(train_data, vocabulary_file, encoded_vocabulary_file)
print('Done.')


print('Saving encoded dictionary ...')
with open(encoded_vocabulary_file, 'w') as encoded_vocab_fid:
    json.dump(encoded_vocab, encoded_vocab_fid)

print('Encoding captions & Building complete json data ...')
for idd, split_data in enumerate(data):
    ids = split_data.keys()
    for idx in ids:
        split_data[idx]['encoded_sentences'] = [encode_caption(sent, encoded_vocab) for sent in split_data[idx]['sentences']]


print('Writing json data ...')
splits = ['train', 'val', 'test']
for idd, split_data in enumerate(data):
    with open(os.path.join(save_root, splits[idd]+'.json'), 'w') as fid:
        json.dump(split_data, fid)


print('Building Glove word embedding numpy array ...')
glove_embed_size = 300
glove_vocab = np.zeros(shape=(len(encoded_vocab.keys()), glove_embed_size))
glove_dict = {}
with open(glove_word_embed_path, 'r') as fid:
    lines = fid.readlines()
    for line in lines:
        items = line.strip().split(' ')
        assert(len(items) == glove_embed_size+1)
        word = items[0]
        embed = np.array([float(val) for val in items[1:]])
        glove_dict[word] = embed

print('All Glove words: {}'.format(len(glove_dict.keys())))

count = 0
for word in encoded_vocab.keys():
    if word in glove_dict.keys():
        glove_vocab[encoded_vocab[word]] = glove_dict[word]
    elif stem(word) in glove_dict.keys():
        glove_vocab[encoded_vocab[word]] = glove_dict[stem(word)]
    else:
        glove_vocab[encoded_vocab[word]] = np.zeros(shape=(glove_embed_size,))
        count += 1

print('{} / {} has no corresponding Glove word embedding.'.format(count, len(encoded_vocab.keys())))
print('Writing Glove word embeddings ...')
np.save(os.path.join(save_root, 'charades_glove_embeds.npy'), glove_vocab)

end_time = time.time()

print('Total running time: %f seconds.'%(end_time - start_time))