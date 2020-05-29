import torch
import numpy as np
from language_models_utils import idx2word
from scipy import stats

def cosine(v, t ):
    """
    Pair-wise cosine vector with matrix's row-vectors
    :param v:  vector (torch tensor)
    :param t: matrix (torch tensor))
    :return: pair-wise cosine scores vector (torch tensor)
    """

    if len(v.shape) == 1:
        v = v.unsqueeze(0)
    v = v.repeat(len(t), 1)
    sims = torch.nn.functional.cosine_similarity(v,t)
    return sims


def nearest_neighbors(vector, word_emb_matrix, vocab, n=10, with_scores=False, into_words = False, model_type = 'LSTM', cosine_sim = True):
    '''
    Get nearest neighbors of a vector wrt word embedding matrix
    :param vector: vector (torch tensor)
    :param word_emb_matrix: word embedding matrix (torch tensor)
    :param vocab: vocabulary
    :param n: number of neighbors; if 'all' gives back all scores
    :param with_scores: return also similarity scores
    :into_words: return words as forms and not as indices
    :return: if with_scores is True: tuple (list of neighbors , list of scores); else list of neighbors
    '''
    if cosine_sim:
        # Calculate cosine similarity scores
        scores = cosine(vector, word_emb_matrix).squeeze(0)
    else:
        #Calculate dot product scores (through matrix multiplication with embedding matrix)
        scores = torch.matmul(vector, word_emb_matrix.t()).squeeze(0)
    if n == 'all':
        n = len(vocab.idx2word)
    scores = scores.cpu().detach().numpy()
    #Sort words by similarity
    nns = list(np.argsort(scores, axis=0))[::-1][:n]
    scores = [scores[i] for i in nns]
    if into_words:
        # Return words instead of their indices
        nns = [idx2word(i, vocab, model_type) for i in nns]
    if with_scores:
        return nns, scores
    else:
        return nns


def cosine_similarity(vec1, vec2):
    '''
    Cosine similarity between two vectors
    :param vec1: vector (torch tensor)
    :param vec2: vector (torch tensor)
    :return: cosine score (float)
    '''
    return float(torch.nn.functional.cosine_similarity(vec1.unsqueeze(0),vec2.unsqueeze(0)))


def dot_product(vec1, vec2):
    '''
    Dot product between two vectors
    :param vec1: vector (torch tensor)
    :param vec2: vector (torch tensor)
    :return: cosine score (float)
    '''
    dot = float(torch.dot(vec1.squeeze(0), vec2.squeeze(0)).detach().cpu().numpy())
    return dot



