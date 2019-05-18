import torch
import torch.nn as nn
import torch.nn.functional as F
from .original_encoder import OriginalEncoder
from .paraphrase_encoder import ParaEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        self.original_encoder = OriginalEncoder(self.params)
        self.paraphrase_encoder = ParaEncoder(self.params)

    def forward(self, original_input, paraphrase_input, ):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentences with shape of [batch_size, latent_variable_size]
        # """
        # [_, seq_len, embed_size] = original.size()
        # original_encoder_hidden = self.original_encoder.init_hidden()
        #
        # for original_input, paraphrase_input in zip(original, paraphrase):
        original_encoder_hidden = self.original_encoder(torch.reshape(original_input, (1, seq_len, embed_size)),
                                                        original_encoder_hidden)
        paraphrase_final_state = self.paraphrase_encoder(torch.reshape(paraphrase_input, (1, seq_len, embed_size)),
                                                             original_encoder_hidden)

        return paraphrase_final_state

    def init_hidden(self):
        return torch.zeros(2*self.params.encoder_num_layers,
                           1, self.params.encoder_rnn_size, device=device)
