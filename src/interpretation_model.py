import torch
import numpy as np
from language_models_utils import encode_context, get_word_embedding
from torch.autograd import Variable

def get_expected_and_lexical_vectors(language_model, vocab, context, index_target, model_type = 'LSTM', cuda = False):
    '''
    Extract expected vector and lexical vector
    Expected vector: last predictive hidden state + transformation
    Lexical vector: word embedding
    :param language_model: language model
    :param vocab: model's vocabulary
    :param word_emb_matrix: word embedding matrix
    :param context: context string
    :param index_target: position index of target word
    :param model_type: model type string
    :param cuda: if True, use GPU
    :return: expected and lexical vectors tuple (torch tensor)
    '''
    # Embed context in torch tensor , get new target index, and word identity
    data_context, index_target, word = encode_context(context, index_target, vocab, cuda=cuda, model_type= model_type, masked = True)
    # Lexical information vector = Word embedding
    lexical_vector = get_word_embedding(word, vocab, language_model, model_type)
    if model_type == 'LSTM':
        # Run LSTM on context and extract predictive hidden states (processing left and right context of the word, but not word itself)
        hidden = language_model.init_hidden(1)
        predictive_hidden_layers, _ = language_model.extract_hidden_layers(data_context, hidden, index_target)
        #Get last hiddenstate
        last_state = predictive_hidden_layers[-1]
        T = torch.nn.Sequential(language_model.decoder.linear, language_model.decoder.nonlinearity)
    elif model_type.startswith('BERT'):
        #Get predictive hidden states (with target masked)
        _, all_hidden_states = language_model(data_context)
        # Get last hidden state
        last_state = torch.Tensor(all_hidden_states[-1].squeeze(0)[index_target].cpu().detach().numpy()[0])
        if cuda: last_state.cuda()
        T = language_model.cls.predictions.transform
    # apply transformation to last state
    expected_vector = T(last_state)
    return expected_vector, lexical_vector


def combine_expected_and_lexical(expected, lexical, combination_type ='avg', normalize=True, alpha_param = 0.5, cuda = False):
    """
    Return combination of expectation and lexical vectors (avg or delta method)
    :param expected: expected vector (torch tensor)
    :param lexical: lexical vector (torch tensor)
    :param normalize: normalize vector before combining
    :param cuda: if True, use GPU
    :param alpha_param: parameter alpha regulating the combination
    :return: vector (torch tensor)
    """
    if normalize:
        expected = torch.nn.functional.normalize(expected, p=2, dim=0)
        lexical = torch.nn.functional.normalize(lexical, p=2, dim=0)
    if combination_type ==  'avg':
        output = alpha_param * lexical + (1 - alpha_param) * expected
        output = torch.nn.functional.normalize(output, p=2, dim=0)
    elif combination_type == 'delta':
        expectation_tmp = expected.clone()
        word_tmp = lexical.clone()
        # Calculate gradient of distance expected-lexical, wrt expectated
        expectation_tmp = Variable(expectation_tmp.unsqueeze(0), requires_grad=True)
        word_tmp = Variable(word_tmp.unsqueeze(0), requires_grad=False)
        distance_function = torch.nn.CosineEmbeddingLoss()
        target = torch.ones(1)
        if cuda: target.cuda()
        # Calculate distance and its gradients
        distance = distance_function(expectation_tmp, word_tmp, target)
        distance.backward()
        grad_exp = expectation_tmp.grad.clone()
        #Substract gradient to expected vector
        updated_exp = expectation_tmp - alpha_param * grad_exp
        output = updated_exp.squeeze(0)
    if normalize:
        output = torch.nn.functional.normalize(output, p=2, dim=0)
    return output