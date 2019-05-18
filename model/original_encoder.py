import torch
import torch.nn as nn
import torch.nn.functional as F

from selfModules.highway import Highway
from utils.functional import parameters_allocation_check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OriginalEncoder(nn.Module):
    def __init__(self, params):
        super(OriginalEncoder, self).__init__()

        self.params = params
        self.bidirectional = self.params.bidirectional
        self.hw1 = Highway(self.params.sum_depth + self.params.word_embed_size, 2, F.relu)

        self.rnn = nn.LSTM(input_size=self.params.word_embed_size + self.params.sum_depth,
                           hidden_size=self.params.encoder_rnn_size,
                           num_layers=self.params.encoder_num_layers,
                           batch_first=True,
                           bidirectional=self.bidirectional)
        self.mul_factor = 2
        if not self.bidirectional:
            self.mul_factor = 1

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentences with shape of [batch_size, latent_variable_size]
        """

        [batch_size, seq_len, embed_size] = input.size()

        input = input.view(-1, embed_size)
        input = self.hw1(input)
        input = input.view(batch_size, seq_len, embed_size)

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        ''' Unfold rnn with zero initial state and get its final state from the last layer
        '''
        out, final_state = self.rnn(input)

        return final_state

    def init_hidden(self, batch_size):
        return tuple([torch.zeros(self.mul_factor * self.params.encoder_num_layers,
                                  batch_size, self.params.encoder_rnn_size, device=device)] * 2)
