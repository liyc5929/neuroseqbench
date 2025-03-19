import math
import warnings
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.benchmark.framework.utils.criterion.IM_loss import Distrloss_layer

# from src.benchmark.framework.network.architecture import MergeDimension, SplitDimension
from src.benchmark.framework.utils.tools import reset_states
from src.benchmark.framework.network.architecture.layer import BatchNorm1d, ThresholdDependentBatchNorm1d, \
    TemporalEffectiveBatchNorm1d, GradwithTrace
from src.benchmark.framework.network.architecture.tcn import TemporalConvNet
from src.benchmark.framework.network.architecture.rnn import script_lstm


# class Decoder(nn.Module):
#     def __init__(self,
#                  ninp: int,
#                  ntokens: int,
#                  ):
#         super(Decoder, self).__init__()
#         self.ninp = ninp
#         self.nout = ntokens
#         # word embedding decoder
#         self.decoder = nn.Linear(self.ninp, self.nout)
#         nn.init.zeros_(self.decoder.bias)
#
#     def forward(self, x):
#         x = x.view(-1, x.size(-1))
#         x = self.decoder(x)
#         return x

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class LMSNN(nn.Module):
    def __init__(self,
                 rnn_type,
                 nlayers,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 dropout_words,
                 dropout_embedding,
                 dropout_forward,
                 dropout,
                 spiking_neuron=None,
                 bn=None,
                 loss=None,
                 args=None,
                 ):
        super(LMSNN, self).__init__()

        self.args = args

        # language model specifics
        self.nlayers = nlayers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.IM_Loss = (loss == 'IM')
        self.ASGL_Loss = (loss == 'ASGL')
        if self.IM_Loss:
            self.distrloss_layers = []

        # dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_forward = dropout_forward
        self.dropout = dropout

        # input and output layers
        self.locked_dropout = LockedDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)
        self.return_state = True

        self.init_weights(initrange=0.1)

        # Tie weights of embedding and decoder
        self.decoder.weight = self.embeddings.weight

        # build BN
        if bn == 'bn':
            bns = [BatchNorm1d(emb_dim if l == nlayers - 1 else hidden_dim) for l in range(nlayers)]
        elif bn == 'ln':
            bns = [nn.LayerNorm(emb_dim if l == nlayers - 1 else hidden_dim) for l in range(nlayers)]
        elif bn == 'tdbn':
            bns = [ThresholdDependentBatchNorm1d(alpha=1, v_th=spiking_neuron().threshold,
                                                 num_features=emb_dim if l == nlayers - 1 else hidden_dim) for l in
                   range(nlayers)]
        elif bn == 'tebn':
            bns = [TemporalEffectiveBatchNorm1d(T=spiking_neuron().time_step,
                                                num_features=emb_dim if l == nlayers - 1 else hidden_dim) for l in
                   range(nlayers)]
        elif bn is None:
            bns = [None for _ in range(nlayers)]
        else:
            raise NotImplementedError

        # RNN model definition
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnns = [nn.LSTM(emb_dim if l == 0 else hidden_dim,
                                 emb_dim if l == nlayers - 1 else hidden_dim,
                                 num_layers=1, batch_first=False, dropout=0)
                         for l in range(nlayers)]
        elif rnn_type == 'gru':
            self.rnns = [nn.GRU(emb_dim if l == 0 else hidden_dim,
                                emb_dim if l == nlayers - 1 else hidden_dim,
                                num_layers=1, batch_first=False, dropout=0)
                         for l in range(nlayers)]
        elif rnn_type in ['lif', 'rlif', 'plif', 'glif', 'alif', 'clif', 'celif', 'tclif', 'psn', 'spsn', 'stclif',
                          'lmh', 'adlif']:
            self.linears = [nn.Linear(emb_dim if l == 0 else hidden_dim,
                                      emb_dim if l == nlayers - 1 else hidden_dim,
                                      )
                            for l in range(nlayers)]
            if bn is None:
                self.snns = [spiking_neuron(neuron_num=emb_dim if l == nlayers - 1 else hidden_dim) for l in
                             range(nlayers)]
            else:
                self.snns = [
                    spiking_neuron(neuron_num=emb_dim if l == nlayers - 1 else hidden_dim, bn=bns[l]) for
                    l in range(nlayers)]

        elif rnn_type == 'dhsnn':
            self.snns = [spiking_neuron(input_features=emb_dim if l == 0 else hidden_dim,
                                        neuron_num=emb_dim if l == nlayers - 1 else hidden_dim) for l in range(nlayers)]
        else:
            raise NotImplementedError(f"Model '{rnn_type}' not implemented.")
        if rnn_type in ['lif', 'rlif', 'plif', 'glif', 'alif', 'clif', 'celif', 'tclif', 'psn', 'spsn', 'stclif', 'lmh',
                        'adlif']:
            self.linears = nn.ModuleList(self.linears)
            self.snns = nn.ModuleList(self.snns)
        elif rnn_type == 'dhsnn':
            self.snns = nn.ModuleList(self.snns)
        else:
            self.rnns = nn.ModuleList(self.rnns)
        if self.IM_Loss:
            for l in range(nlayers):
                self.distrloss_layers.append(Distrloss_layer(thresh=0.5))

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return [(weight.new_zeros(1, batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(1, batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'gru':
            return [weight.new_zeros(1, batch_size,
                                     self.emb_dim if l == self.nlayers - 1 else self.hidden_dim)
                    for l in range(self.nlayers)]
        elif self.rnn_type in {'lif', 'rlif', 'plif', 'glif', 'stclif', 'psn', 'spsn'}:
            return [(weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim))  # v, y
                    for l in range(self.nlayers)]
        elif self.rnn_type in {'alif'}:
            return [(weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_full((batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim), 0.01)  # v, y, b
                     )
                    for l in range(self.nlayers)]
        elif self.rnn_type in {'celif'}:
            return [(weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_full((batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim), 0.5)  # v, y, thresh
                     )
                    for l in range(self.nlayers)]
        elif self.rnn_type in {'lmh'}:
            return [(weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_full((batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim), 0.25),
                     weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim)  # vd, vs, y
                     )
                    for l in range(self.nlayers)]
        elif self.rnn_type in {'dhsnn'}:
            branch = self.snns[0].branch
            return [(weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim, branch)  # v, y, d
                     )
                    for l in range(self.nlayers)]
        elif self.rnn_type in {'clif', 'tclif', 'adlif'}:
            return [(weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim)
                     # clif: u,y,m; tclif: v1,v2,y; adlif: v y wt
                     )
                    for l in range(self.nlayers)]
        else:
            raise NotImplementedError(
                f"Model '{self.rnn_type}' not implemented.")

    def forward(self, inputs, state):
        if self.rnn_type not in ['gru', 'lstm']:
            return self.spk_forward(inputs, state)
        # embedding forward
        embedded = embedded_dropout(self.embeddings, inputs,
                                    dropout=self.dropout_words if self.training else 0)

        embedded = self.locked_dropout(
            embedded, dropout=self.dropout_embedding)

        # rnn forward
        new_states = []
        hiddens = embedded
        for l, rnn in enumerate(self.rnns):
            hiddens, final_states = rnn(hiddens, state[l])

            new_states.append(final_states)

            if l != self.nlayers - 1:
                hiddens = self.locked_dropout(
                    hiddens, dropout=self.dropout_forward)

        # decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)

        decoded = self.decoder(hiddens)
        return decoded, new_states

    def spk_forward(self, inputs, state):  # inputs: [T, B, N]
        embedded = embedded_dropout(self.embeddings, inputs,
                                    dropout=self.dropout_words if self.training else 0)

        embedded = self.locked_dropout(
            embedded, dropout=self.dropout_embedding)

        # rnn forward
        new_states = []
        hiddens = embedded
        self.loss = []
        if self.rnn_type == 'dhsnn':
            for l, (snn) in enumerate(self.snns):
                hiddens, final_states = snn(hiddens, state[l])
                new_states.append(final_states)
                if self.IM_Loss:
                    self.loss.append(self.distrloss_layers[l](hiddens))
                if l != self.nlayers - 1:
                    hiddens = self.locked_dropout(
                        hiddens, dropout=self.dropout_forward)

        else:
            for l, (linear, snn) in enumerate(zip(self.linears, self.snns)):
                if self.training and (self.args.learning_rule == 'eprop'):
                    t_trace = []
                    trace = torch.zeros_like(hiddens[0].detach())
                    for each_step_hidden in hiddens:
                        trace = self.args.decay * trace.detach() + each_step_hidden
                        t_trace.append(trace)
                    t_trace = torch.stack(t_trace)
                    t_trace_output = linear(t_trace)
                    hiddens = linear(hiddens.detach()).detach() + t_trace_output - t_trace_output.detach()
                else:
                    hiddens = linear(hiddens)
                hiddens, final_states = snn(hiddens, state[l])
                new_states.append(final_states)
                if self.IM_Loss:
                    self.loss.append(self.distrloss_layers[l](hiddens))
                if l != self.nlayers - 1:
                    hiddens = self.locked_dropout(
                        hiddens, dropout=self.dropout_forward)

        # decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)

        decoded = self.decoder(hiddens)
        # print('decoded', decoded.mean())
        if self.IM_Loss:
            disloss = (sum([ele for ele in self.loss])) / len(self.loss)
            return decoded, new_states, disloss
        else:
            return decoded, new_states


class LMTCN(nn.Module):
    def __init__(self,
                 emb_dim,
                 vocab_size,
                 dropout_words,
                 dropout_embedding,
                 dropout_forward,
                 dropout,
                 num_channels,
                 kernel_size,
                 spiking_neuron=None,
                 t_internal=4
                 ):
        super(LMTCN, self).__init__()

        # language model specifics
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        # dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_forward = dropout_forward
        self.dropout = dropout

        # input and output layers
        self.locked_dropout = LockedDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)
        self.return_state = True
        self.t_internal = t_internal
        self.init_weights(initrange=0.1)

        # Tie weights of embedding and decoder
        self.decoder.weight = self.embeddings.weight

        self.tcn = TemporalConvNet(emb_dim, num_channels, kernel_size, dropout=dropout_forward,
                                   spiking_neuron=spiking_neuron)
        assert num_channels[-1] == emb_dim

        self.spiking = spiking_neuron is not None

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return [(weight.new_zeros(1, batch_size,
                                  self.emb_dim),
                 weight.new_zeros(1, batch_size,
                                  self.emb_dim))
                for l in range(1)]

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs, state):

        # embedding forward
        embedded = embedded_dropout(self.embeddings, inputs,
                                    dropout=self.dropout_words if self.training else 0)

        embedded = self.locked_dropout(
            embedded, dropout=self.dropout_embedding)

        # rnn forward
        new_states = []
        hiddens = embedded
        # print(hiddens.size())
        # TCN here
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        hiddens = hiddens.permute(1, 2, 0).contiguous()  # [T, B, N] - >[B， N T, ]
        if self.spiking:
            reset_states(self)
            T = self.t_internal
            for t in range(T):
                if t == 0:
                    output = self.tcn(hiddens)
                else:
                    output = output + self.tcn(hiddens)
            hiddens = output / T
        else:
            hiddens = self.tcn(hiddens)
        hiddens = hiddens.permute(2, 0, 1).contiguous()
        # decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)

        decoded = self.decoder(hiddens)
        return decoded, state


class LMLSTM(nn.Module):
    def __init__(self,
                 rnn_type,
                 nlayers,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 dropout_words,
                 dropout_embedding,
                 dropout_forward,
                 dropout,
                 spiking_neuron=None,
                 spiking=False
                 ):
        super(LMLSTM, self).__init__()

        # language model specifics
        self.nlayers = nlayers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_forward = dropout_forward
        self.dropout = dropout

        # input and output layers
        self.locked_dropout = LockedDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)

        self.init_weights(initrange=0.1)

        # Tie weights of embedding and decoder
        self.decoder.weight = self.embeddings.weight

        self.spiking = spiking
        if self.spiking:
            self.spiking_neuron = spiking_neuron()

        # RNN model definition
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            if self.spiking:
                self.rnns = [script_lstm(emb_dim if l == 0 else hidden_dim,
                                         emb_dim if l == nlayers - 1 else hidden_dim,
                                         num_layers=1, batch_first=False, spiking=self.spiking)
                             for l in range(nlayers)]
            else:
                self.rnns = [nn.LSTM(emb_dim if l == 0 else hidden_dim,
                                         emb_dim if l == nlayers - 1 else hidden_dim,
                                         num_layers=1, batch_first=False)
                             for l in range(nlayers)]

        elif rnn_type == 'gru':
            self.rnns = [script_lstm(emb_dim if l == 0 else hidden_dim,
                                     emb_dim if l == nlayers - 1 else hidden_dim,
                                     num_layers=1, batch_first=False, spiking=self.spiking, GRU=True)
                         for l in range(nlayers)
                         ]
        else:
            raise NotImplementedError(f"Model '{rnn_type}' not implemented.")

        self.rnns = nn.ModuleList(self.rnns)

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            if self.spiking:
                return [[(weight.new_zeros(batch_size,
                                           self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                          weight.new_zeros(batch_size,
                                           self.emb_dim if l == self.nlayers - 1 else self.hidden_dim))]
                        for l in range(self.nlayers)]
            else:
                return [(weight.new_zeros(1, batch_size,
                                          self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                         weight.new_zeros(1, batch_size,
                                          self.emb_dim if l == self.nlayers - 1 else self.hidden_dim))
                        for l in range(self.nlayers)]
        elif self.rnn_type == 'gru':
            if self.spiking:
                return [[(weight.new_zeros(batch_size,
                                           self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                          weight.new_zeros(batch_size,
                                           self.emb_dim if l == self.nlayers - 1 else self.hidden_dim))]
                        for l in range(self.nlayers)]
            else:
                return [[weight.new_zeros(batch_size,
                                          self.emb_dim if l == self.nlayers - 1 else self.hidden_dim)]
                        for l in range(self.nlayers)]
        else:
            raise NotImplementedError(
                f"Model '{self.rnn_type}' not implemented.")

    def forward(self, inputs, state):

        # embedding forward
        embedded = embedded_dropout(self.embeddings, inputs,
                                    dropout=self.dropout_words if self.training else 0)

        embedded = self.locked_dropout(
            embedded, dropout=self.dropout_embedding)

        # rnn forward
        new_states = []
        hiddens = embedded
        for l, rnn in enumerate(self.rnns):
            if self.spiking:
                hiddens, final_states = rnn(hiddens, state[l], threshold=self.spiking_neuron.threshold,
                                            surrogate_function=self.spiking_neuron.surrogate_function)
            else:
                hiddens, final_states = rnn(hiddens, state[l])
            new_states.append(final_states)

            if l != self.nlayers - 1:
                hiddens = self.locked_dropout(
                    hiddens, dropout=self.dropout_forward)

        # decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)

        decoded = self.decoder(hiddens)
        return decoded, new_states


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)
    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)
class LMTransformer(nn.Module):

    def __init__(self,
                 nhead,
                 nlayers,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 dropout_embedding,
                 dropout_forward,
                 spiking_neuron=None,
                 spiking=False):
        super(LMTransformer, self).__init__()
        # emb_dim = 128
        # vocab_size = 10000
        # h = 8
        # n_layers = 3
        # dropout_forward = 0.2
        # dropout_embedding = 0.2
        # tied_weights = True
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout_forward,
                                                )
        encoder_norm = nn.LayerNorm(emb_dim)
        custom_encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                          num_encoder_layers=nlayers, dropout=dropout_forward, custom_encoder=custom_encoder).encoder
        self.decoder = nn.Linear(emb_dim, vocab_size)

        self.emb_dropout = dropout_embedding
        self.init_weights()
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=dropout_embedding)
        self.src_mask = None

        self.decoder.weight = self.embeddings.weight

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, has_mask=True):
        if has_mask:
            device = input.device
            if self.src_mask is None or self.src_mask.size(0) != len(input):
                mask = self._generate_square_subsequent_mask(len(input)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        emb = embedded_dropout(self.embeddings, input,
                               dropout=0 if self.training else 0)

        # input = input.transpose(0, 1)
        # emb = self.embeddings(input)
        # emb = emb.transpose(0, 1)
        emb = self.pos_encoder(emb)
        y = self.transformer(emb, mask=self.src_mask)
        y = self.decoder(y)
        return y

def SDSA3(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        scale=0.125
):
    # assert attn_mask is None
    # assert dropout_p == 0.0
    T, B, Nt, E = q.shape
    # q = q / math.sqrt(E)
    if False:
        attn = k.transpose(-2, -1) @ v
        # print(f"attn: {attn.size()}")
        # if dropout_p > 0.0:
        #     attn = nn.functional.dropout(attn, p=dropout_p)
        output = (q @ attn) * scale
    else:
        # q = q / math.sqrt(E)
        attn = q @ k.transpose(-2, -1)
        attn = attn * attn_mask
        # if dropout_p > 0.0:
        #     attn = nn.functional.dropout(attn, p=dropout_p)
        # attn = nn.functional.softmax(attn, dim=-1)
        # print(f"attn: {attn.size()}")
        # print(f"q: {q.size()}")
        output = (attn @ v) * scale

    # print(f"attn output: {output.size()}")
    return output, attn

def _in_projection_packed(
        q,
        k,
        v,
        w,
        b=None,
):
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return nn.functional.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (nn.functional.linear(q, w_q, b_q),) + nn.functional.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return nn.functional.linear(q, w_q, b_q), nn.functional.linear(k, w_k, b_k), nn.functional.linear(v, w_v, b_v)

def spk_multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim_to_check,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        dropout_p,
        out_proj_weight,
        out_proj_bias,
        q_lif=None,
        k_lif=None,
        v_lif=None,
        attn_lif=None,
        training=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # set up shape vars
    T, tgt_len, bsz, embed_dim = query.shape
    T, src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    q = q_lif(q)
    k = k_lif(k)
    v = v_lif(v)
    # print(f"q: {q.size()}, k: {k.size()}, v: {v.size()}")
    # prep attention mask
    # assert attn_mask is None, "Not implemented!"

    # prep key padding mask
    assert key_padding_mask is None, "Not implemented"

    # add bias along batch dimension (currently second)

    assert bias_k is None
    assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(T, tgt_len, bsz * num_heads, head_dim).transpose(1, 2)
    k = k.contiguous().view(T, k.shape[1], bsz * num_heads, head_dim).transpose(1, 2)
    v = v.contiguous().view(T, v.shape[1], bsz * num_heads, head_dim).transpose(1, 2)
    # print(f"****q: {q.size()}, k: {k.size()}, v: {v.size()}")

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")




    # add zero attention along batch dimension (now first)

    # update source sequence length after adjustments
    src_len = k.size(2)




    # merge key padding and attention masks
    assert key_padding_mask is None

    # convert mask to float

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = SDSA3(q, k, v, attn_mask, dropout_p)
    # print(f"attn_output: {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous().view(T, tgt_len, bsz, embed_dim)
    attn_output = attn_lif(attn_output)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(T, bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=2) / num_heads
    else:
        return attn_output, None


class SpkMultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']

    # bias_k: Optional[torch.Tensor]
    # bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 batch_first=False, spiking_neuron=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpkMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.bias_k = self.bias_v = None

        self._reset_parameters()

        assert spiking_neuron is not None
        self.scale = 0.125
        self.head_lif = spiking_neuron()
        self.q_lif = spiking_neuron()
        self.k_lif = spiking_neuron()
        self.v_lif = spiking_neuron()
        self.attn_lif = spiking_neuron()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    # def __setstate__(self, state):
    #     # Support loading old MultiheadAttention checkpoints generated by v1.1.0
    #     if '_qkv_same_embed_dim' not in state:
    #         state['_qkv_same_embed_dim'] = True
    #
    #     super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """
        assert self.batch_first is False
        # if self.batch_first:
        #     query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = spk_multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            q_lif=self.q_lif, k_lif=self.k_lif, v_lif=self.v_lif, attn_lif=self.attn_lif,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class SpkTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=torch.nn.functional.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, spiking_neuron=None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpkTransformerEncoderLayer, self).__init__()
        self.self_attn = SpkMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, spiking_neuron=spiking_neuron,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.head_lif = spiking_neuron()
        self.fc1_lif = spiking_neuron()
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.fc2_lif = spiking_neuron()
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        # if isinstance(activation, str):
        #     self.activation = _get_activation_fn(activation)
        # else:
        self.activation = activation

    # def __setstate__(self, state):
    #     if 'activation' not in state:
    #         state['activation'] = F.relu
    #     super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = self.head_lif(x)
        # print(f"head_lif: {x.size()}")
        assert self.norm_first is False
        # if self.norm_first:
        #     x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        #     x = x + self._ff_block(self.norm2(x))
        # else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        # print(f"norm1: {x.size()}")
        x = self.norm2(x + self._ff_block(x))
        # print(f"norm2: {x.size()}")

        return x

    # self-attention block
    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.fc1_lif(x)
        x = self.linear1(x)
        x = self.fc2_lif(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class LMSpkTransformer(nn.Module):

    def __init__(self,
                 nhead,
                 nlayers,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 dropout_embedding,
                 dropout_forward,
                 spiking_neuron=None,
                 T=4):
        super(LMSpkTransformer, self).__init__()
        # emb_dim = 128
        # vocab_size = 10000
        # h = 8
        # n_layers = 3
        # dropout_forward = 0.2
        # dropout_embedding = 0.2
        # tied_weights = True
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        encoder_layer = SpkTransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout_forward, spiking_neuron=spiking_neuron
                                                )
        encoder_norm = nn.LayerNorm(emb_dim)
        custom_encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                          num_encoder_layers=nlayers, dropout=dropout_forward, custom_encoder=custom_encoder).encoder
        self.decoder = nn.Linear(emb_dim, vocab_size)
        self.linear_spk = spiking_neuron()
        self.time_window = T
        self.emb_dropout = dropout_embedding
        self.init_weights()
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=dropout_embedding)
        self.src_mask = None

        self.decoder.weight = self.embeddings.weight

    def _generate_square_subsequent_mask(self, sz):
        return torch.tril(torch.ones(sz, sz))

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, has_mask=True):
        if has_mask:
            device = input.device
            if self.src_mask is None or self.src_mask.size(0) != len(input):
                mask = self._generate_square_subsequent_mask(len(input)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        emb = embedded_dropout(self.embeddings, input,
                               dropout=0 if self.training else 0)

        # input = input.transpose(0, 1)
        # emb = self.embeddings(input)
        # emb = emb.transpose(0, 1)
        emb = self.pos_encoder(emb)
        emb = (emb.unsqueeze(0)).repeat(self.time_window, 1, 1, 1)

        y = self.transformer(emb, mask=self.src_mask)
        y = self.linear_spk(y)
        y = self.decoder(y)
        return y.mean(0)

class LMTransformer2(nn.Transformer):
    def __init__(self,
                 nhead,
                 nlayers,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 dropout_words,
                 dropout_embedding,
                 dropout_forward,
                 dropout,
                 spiking_neuron=None,
                 spiking=False
                 ):
        super(LMTransformer2, self).__init__(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                             num_encoder_layers=nlayers, dropout=dropout_forward)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=0.)

        # language model specifics
        self.nlayers = nlayers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_forward = dropout_forward
        self.dropout = dropout

        # input and output layers
        self.locked_dropout = LockedDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)

        self.init_weights(initrange=0.1)

        # Tie weights of embedding and decoder
        self.decoder.weight = self.embeddings.weight

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs, has_mask=False):
        if has_mask:
            device = inputs.device
            if self.src_mask is None or self.src_mask.size(0) != len(inputs):
                mask = self._generate_square_subsequent_mask(len(inputs)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        # embedding forward
        embedded = embedded_dropout(self.embeddings, inputs,
                                    dropout=0 if self.training else 0)
        # embedded = self.pos_encoder(embedded)
        embedded = self.locked_dropout(
            embedded, dropout=self.dropout_embedding)

        # rnn forward
        new_states = []
        hiddens = embedded
        hiddens = self.encoder(hiddens, mask=self.src_mask)
        # decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)
        hiddens = hiddens.transpose(0, 1)
        decoded = self.decoder(hiddens)
        return decoded
