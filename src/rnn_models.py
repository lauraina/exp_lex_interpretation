
import torch.nn as nn
import torch
from torch.autograd import Variable

class RNNLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, word2idx, emb_size, hidden_sizes, dropout,
                 rnn_type="LSTM", pretrained_embs=None, fixed_embs=False, tied=None):
        super(RNNLanguageModel, self).__init__()

        self.encoder = Encoder(word2idx, emb_size, pretrained_embs, fixed_embs)
        self.decoder = Decoder(len(word2idx), hidden_sizes[-1], tied, self.encoder)

        self.rnn = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        seq_len = output.size(0)
        batch_size = output.size(1)
        output = output.view(output.size(0) * output.size(1), output.size(2))

        decoded = self.decoder(output)
        return decoded.view(seq_len, batch_size, decoded.size(1)), hidden

    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)


class BidirectionalLanguageModel(nn.Module):
    """Container module with a (shared) encoder, two recurrent modules -- forward and backward -- and a decoder."""

    def __init__(self, word2idx, emb_size, hidden_sizes, dropout,
                 rnn_type="LSTM", pretrained_embs=None, fixed_embs=False, tied=None):
        super(BidirectionalLanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.encoder = Encoder(word2idx, emb_size, pretrained_embs, fixed_embs)
        self.decoder = Decoder(len(word2idx), hidden_sizes[-1], tied, self.encoder)

        self.forward_lstm = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)
        self.backward_lstm = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)

        self.rnn_type = rnn_type
        self.hidden_sizes = hidden_sizes
        self.nlayers = len(hidden_sizes)

    def forward(self, input, hidden):
        input_f, input_b = input
        emb_f = self.drop(self.encoder(input_f))
        emb_b = self.drop(self.encoder(input_b))

        hidden_f = hidden[0]
        hidden_b = hidden[1]

        output_f, hidden_f = self.forward_lstm(emb_f, hidden_f)
        output_b, hidden_b = self.backward_lstm(emb_b, hidden_b)

        output = output_f + flip(output_b, dim=0)   # output is sum of forward and backward

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), (hidden_f, hidden_b), output.data

    def init_hidden(self, bsz):
        return self.forward_lstm.init_hidden(bsz), self.backward_lstm.init_hidden(bsz)


    def extract_hidden_layers(self, input, hidden, index_target, concat = False):

        input_f, input_b = input

        index_target_f = index_target
        index_target_b = len(input_f) - index_target + 1

        emb_f = self.drop(self.encoder(input_f[:index_target_f]))
        emb_b = self.drop(self.encoder(input_b[:index_target_b]))

        hidden_f = hidden[0]
        hidden_b = hidden[1]

        output_f, hidden_f = self.forward_lstm(emb_f, hidden_f)
        output_b, hidden_b = self.backward_lstm(emb_b, hidden_b)

        predictive_hidden_layers = []
        for i in range(len(hidden_f)):
            f = hidden_f[i][0]
            b = flip(hidden_b[i][0], dim=0)
            if concat:
                output_i =  torch.cat((f, b), dim = 2)
            else:
                output_i = f + b
            predictive_hidden_layers.append(output_i.squeeze(0).squeeze(0))

        hidden = self.init_hidden(1)

        emb_f = self.drop(self.encoder(input_f[:index_target_f + 1 ]))
        emb_b = self.drop(self.encoder(input_b[:index_target_b + 1 ]))

        hidden_f = hidden[0]
        hidden_b = hidden[1]

        output_f, hidden_f = self.forward_lstm(emb_f, hidden_f)
        output_b, hidden_b = self.backward_lstm(emb_b, hidden_b)

        current_hidden_layers = []
        for i in range(len(hidden_f)):
            f = hidden_f[i][0]
            b = flip(hidden_b[i][0], dim=0)
            if concat:
                output_i =  torch.cat((f, b), dim = 2)
            else:
                output_i = f + b
            current_hidden_layers.append(output_i.squeeze(0).squeeze(0))

        return predictive_hidden_layers, current_hidden_layers

def batchify(data, bsz, cuda=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


def get_batch(source, i, bptt, mode, evaluation):
    if mode == "forward":
        return _get_batch(source, i, bptt, evaluation)
    elif mode == "backward":
        return _get_batch(flip(source, dim=0), i, bptt, evaluation)
    elif "bidir" in mode:
        return _get_batch_bidirectional(source, i, bptt, evaluation)


def _get_batch(source, i, bptt, evaluation):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], requires_grad=False)  # Here it starts from one
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1),
                                                    ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def _get_batch_bidirectional(data, i, seq_len, evaluation):
    seq_len = min(seq_len, len(data) - 2 - i)
    data_f, targets = _get_batch(data, i, seq_len, evaluation)
    data_b = Variable(data[i + 2:i+seq_len + 2], requires_grad=False)
    data_b = flip(data_b, dim=0)
    return (data_f, data_b), targets


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def update_hidden(model, mode, hidden, batch_size):
    if mode == "bidir":
        # fixed window length, suboptimal for tokens close to the boundaries
        return model.init_hidden(batch_size)
    elif mode == "bidir_cont": # forward continuous
        hidden1 = repackage_hidden(hidden)
        hidden2 = model.init_hidden(batch_size)
        # keep forward, restart backward
        return hidden1, hidden2
    elif mode == "forward" or mode == "backward":
        return repackage_hidden(hidden)    # continuous hidden state




