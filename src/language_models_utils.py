import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from transformers import *
from rnn_models import batchify, get_batch
import string
import numpy as np

def load_languagemodel(model_type, model_dir = '..', cuda = False):
    """
    Load language model, word embedding matrix and vocabulary both for LSTM and BERT MODEL
    :param model_type: type of model string (LSTM or BERT-base, BERT-large)
    :model_dir: path to directory containing the model
    :param cuda: if True: use GPU
    :return: language model object and vocabulary object
    """
    if model_type == 'LSTM':
        seed = 1111
        language_model = model_dir + 'biLSTM_tiedT.pt'
        vocab = model_dir + 'biLSTM_vocab.txt'
        if cuda:
            torch.cuda.manual_seed(seed)
        with open(language_model, 'rb') as f:
            language_model = torch.load(f, map_location=lambda storage, loc: storage)
        if cuda:
            language_model.cuda()
        vocab = LSTMVocabulary(vocab)  # Load vocabulary
        word_emb_matrix =  language_model.encoder.embedding.weight.data

    elif model_type.startswith('BERT'):
        if model_type == 'BERT-base':
            MODEL = (BertForMaskedLM, BertTokenizer, 'bert-base-cased')
        elif model_type == 'BERT-large':
            MODEL = (BertForMaskedLM, BertTokenizer, 'bert-large-cased-whole-word-masking')
        model_class, tokenizer_class, pretrained_weights = MODEL
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        language_model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        if cuda: language_model.cuda()
        vocab = BERTVocabulary(tokenizer)
        word_emb_matrix =  language_model.bert.embeddings.word_embeddings.weight.data

    return language_model, word_emb_matrix, vocab


def idx2word(idx, vocab, model_type = 'LSTM'):
    """
    Shortcuts: from id to word in vocab
    :param idx: index of word in vocabulary (int)
    :param vocab: vocabulary of the model
    :param model: model
    :return: word with given index in vocabulary (string)
    """

    if model_type.startswith('LSTM'):
        return vocab.idx2word[idx]
    elif model_type.startswith('BERT'):
        return vocab.idx2piece[idx]


def word2idx(word, vocab, model_type = 'LSTM'):
    '''
    Shortcuts: from word to id in vocab
    :param word: word
    :param vocab: vocabulary of the model
    :param model:  type of model string (LSTM or BERT-base, BERT-large)
    :return: index of word in the vocabulary (int)
    '''
    if in_vocab(word, vocab, model_type):
        if model_type.startswith('LSTM'):
            return vocab.word2idx[word]
        elif model_type.startswith('BERT'):
            return vocab.piece2idx[word][0]
    else:
        print("This word is not in the LM's vocabulary.")
        return None

def in_vocab(word, vocab, model_type = 'LSTM'):
    """
    Check if word in vocabulary
    :param word:
    :param vocab:
    :param model_eval:
    :return: bool (true if the word is in the vocabulary)
    """
    if model_type.startswith('LSTM'):
        return word in vocab.word2idx
    elif model_type.startswith('BERT'):
        return word in vocab.piece2idx


def encode_context(context, index_target, vocab, cuda=False, model_type = 'LSTM', masked = True):
    '''
    Pre-process and encode context, get new indices of target word, turn data into tensor
    :param context: context string
    :param index_target: position index of target word
    :param model_type: type of model string (LSTM or BERT-base, BERT-large)
    :return: list of context words, index of target word, target word
    '''
    if model_type == 'LSTM':
        mode = 'bidir',
        context = context.split()
        # Identify and mark word occurrence
        context[index_target] = '<target> ' + context[index_target]
        context = ' '.join(context)
        sep = '<eos>'
        # Add <eos> symbol
        context = context.replace('.', '. ' + sep + ' ')
        context = ' '.join([sep, context])
        context = context.split()
        context.append('<eos>')
        context.append('<eos>')
        # Find again word occurrence
        index_target = context.index('<target>')
        del context[index_target]
        target = context[index_target]
        data_context = torch.cuda.LongTensor(len(context)) if cuda else torch.LongTensor(len(context))
        token = 0
        # out of vocabulary word --> unk
        for word in context:
            if word not in vocab.word2idx:
                data_context[token] = vocab.word2idx["<unk>"]
            else:
                data_context[token] = vocab.word2idx[word]
            token += 1
        # Turn context into tensor
        data_context = batchify(data_context, 1)
        seq_len = len(data_context)
        data_context, _ = get_batch(data_context, 0, seq_len, mode, evaluation=True)
    elif model_type.startswith('BERT'):
        target = context.split()[index_target]
        context = '[CLS] ' + context + ' [SEP] '
        index_target += 1
        # Get position of word after BERT tokenization
        index_target = get_indices_word_after_encoding_BERT(context, index_target, vocab)[0]
        data_context = vocab.encode(context)
        if masked:
            data_context[index_target_word] = vocab.piece2idx['[MASK]'][0]
        data_context = torch.tensor([data_context]).cuda() if cuda else torch.tensor([data_context])
    return data_context, index_target, target


def get_word_embedding(word, vocab, language_model, model_type = 'LSTM'):
    '''
    Return embedding of a word, if not split into subword units (BERT)
    Word embedding: row-vector in input/output embedding matrix
    :param word: word string
    :param vocab: model's vocabulary
    :param model_type: type of model string (LSTM or BERT-base, BERT-large)
    :param wordemb_matrix: model's embedding matrix
    :
    :return: word vector, or None
    '''
    if model_type == 'LSTM':
        return language_model.encoder.embedding.weight.data[vocab.word2idx[word]]
    elif model_type == 'BERT':
        encoded = vocab.word2idx(word)
        if len(encoded) == 1:
            word_vector = language_model.bert.embeddings.word_embeddings.weight.data[encoded[0]].cpu().numpy()
            return word_vector
        else:
            print(word, 'is split into ', len(encoded), 'subword units.')
            return None

def get_logprobability_word(language_model, context, index_target_word, vocab, cuda = False, masked = True, model_type ='LSTM'):
    '''

    :param language_model: language model
    :param context: context str
    :param index_target_word: index of word in context (split by space)
    :param vocab: vocabulary of the model
    :param cuda: if True: use GPU
    :param masked: if probability of word when masked (BERT)
    :param model_type:  type of model string (LSTM or BERT-base, BERT-large)
    :return: log probability of word (float)
    '''
    data_context, index_target, word = encode_context(context, index_target_word, vocab, cuda=cuda,
                                                      model_type=model_type, masked = masked)
    if model_type == 'LSTM':
        hidden = language_model.init_hidden(1)
        output_scores, _, _= language_model(data_context, hidden)
        output_scores = output_scores[index_target - 1]
        logprob_distr = torch.nn.functional.log_softmax(output_scores, dim=1).squeeze()

    if model_type.startswith('BERT'):
        output_scores, _ = model(data_context)
        output_scores = prediction_scores.squeeze(0)[index_target]
        logprob_distr = torch.nn.functional.log_softmax(output_scores, dim=0).cpu().detach().numpy()
    logprob_distr = logprob_distr.data.cpu().numpy()
    logprob_word = logprob_distr[word2idx(word, vocab, model_type)]
    return logprob_word


class LSTMVocabulary(object):
    '''
    Vocabulary object for LSTM
    '''

    def __init__(self, vocab_file=None):
        # mapping words to indices, and viceversa
        self.word2idx = {}
        self.idx2word = []
        # Getting info from vocabulary file
        if vocab_file:
            with open(vocab_file, "rt") as f:
                for l in f:
                    self.add_word(l.strip())

    def add_word(self, word):
        '''
        Add word to vocabulary
        :param word: word string
        '''
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1


class BERTVocabulary():
    '''
    BERT vocabulary object built around the BERT tokenizer
    '''

    def __init__(self, tokenizer):
        # mapping indices to subwords
        self.idx2piece = list(tokenizer.vocab.keys())
        # mapping subword to indices
        self.piece2idx = {word: tokenizer.encode(word) for word in self.idx2piece}
        # original BERT tokenizer
        self.tokenizer = tokenizer

    def word2idx(self, word):
        # map word to indices (parallel to LSTM vocabulary)
        return self.tokenizer.encode(word)

    def encode(self, text):
        '''
        Turns sequence of words into ids
        :param text:
        :return: list of indices
        '''
        return self.tokenizer.encode(text)

    def decode(self, encoded):
        '''
        Turn sequence of ids into sequence of words
        :param encoded:
        :return: list of tokens
        '''
        return [self.idx2piece[int(i)] for i in encoded]


def merge_subwords_BERT(context):
    '''
    Join sub-word elements (needed to find occurrences of a word in a context)
    The word may be split in subwords, containing ## at the boundaries
    :param context: textspan string
    :return: textspan string with merged subwords
    '''
    merged = []
    for i in context:
        if not i.startswith('##'):
            merged.append(i)
        else:
            merged[-1] += i.replace('##', '')
    return merged




def get_indices_word_after_encoding_BERT(context, index_word, vocab):
    '''
    Get position indices of a word occurrence after the BERT encoding
     (the split into subword units may shift the original position, and/or split the word itself into multiple units)
    :param context: context string
    :param index_word: position index of word in original context
    :param vocab: model's vocabulary
    :return: list of indices of word in encoded context
    '''
    # Identity of word
    word = context.split()[index_word]
    # Get indices of word
    encode_word = vocab.word2idx(word)
    # Encode and decode context: get string of textspan with new tokenization
    encoded_context = vocab.encode(context)
    decoded = vocab.decode(encoded_context)
    # Isolate left and right context in original context
    right_context = ''.join(context.split()[:index_word])
    left_context = ''.join(context.split()[index_word + 1 :])

    # Find occurrences of first subword unit of the word in context
    indices = [index for index, value in enumerate(encoded_context) if value == encode_word[0]]
    found = False
    i = 0
    while found == False and i < len(indices):
        index_tmp = indices[i]
        len_word = len(encode_word)
        #Identify if that occurrence of the first subword is the correct one
        #Condition: it has the correct left and right context (when subword units are merged)
        if right_context == ''.join(merge_subwords_BERT(decoded[:index_tmp])) and left_context == ''.join(merge_subwords_BERT(decoded[index_tmp + len(encode_word):])):
            index_word = index_tmp
            found = True
        else:
            # Consider the next occurrence
            i += 1
    if found == False:
        print('Error in finding word in context')
        return None
    # List of the indices spanning the word occurrence
    indices_target = list(range(index_word, index_word + len_word))
    return indices_target