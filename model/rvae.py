import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .original_encoder import OriginalEncoder
from .paraphrase_encoder import ParaEncoder
from selfModules.embedding import Embedding

from utils.functional import kld_coef, parameters_allocation_check, fold

zero_initialize = None


class RVAE(nn.Module):
    def __init__(self, params):
        super(RVAE, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.original_encoder = OriginalEncoder(self.params)
        self.paraphrase_encoder = ParaEncoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

    def forward(self, drop_prob,
                original_encoder_word_input=None, original_encoder_character_input=None,
                paraphrse_encoder_word_input=None, paraphrse_encoder_character_input=None,
                z=None, initial_state=None):
        """

        Args:
            drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
            original_encoder_word_input:
            original_encoder_character_input:
            paraphrse_encoder_word_input:
            paraphrse_encoder_character_input:
            z: context if sampling is performing
            initial_state: initial state of decoder rnn in order to perform sampling

        Returns:
                unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]

        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                  [original_encoder_word_input, original_encoder_character_input,
                                   paraphrse_encoder_word_input],
                                  True) \
            or (z is not None and paraphrse_encoder_word_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        [batch_size, _] = original_encoder_word_input.size()

        original_encoder_input = self.embedding(original_encoder_word_input, original_encoder_character_input)
        paraphrse_encoder_input = self.embedding(paraphrse_encoder_word_input, paraphrse_encoder_character_input)

        global zero_initialize
        if not zero_initialize:
            original_encoder_hidden = self.original_encoder.init_hidden(batch_size)
            zero_initialize = True

        original_encoder_hidden = self.original_encoder(original_encoder_input, original_encoder_hidden)
        context = self.paraphrase_encoder(paraphrse_encoder_input, original_encoder_hidden)

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = torch.exp(0.5 * logvar)

            z = Variable(torch.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None
        if initial_state is None:
            initial_state = original_encoder_hidden
        out, final_state = self.decoder(paraphrse_encoder_input, z, drop_prob, initial_state)

        return out, final_state, kld

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [Variable(torch.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [original_encoder_word_input, original_encoder_character_input,
                paraphrse_encoder_word_input, paraphrse_encoder_character_input, target] = input

            logits, _, kld = self(dropout,
                                  original_encoder_word_input, original_encoder_character_input,
                                  paraphrse_encoder_word_input, paraphrse_encoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            loss = 79 * cross_entropy + kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, kld_coef(i)

        return train

    def validater(self, batch_loader):
        def validate(batch_size, use_cuda):
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(torch.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [original_encoder_word_input, original_encoder_character_input,
                paraphrse_encoder_word_input, paraphrse_encoder_character_input, target] = input

            logits, _, kld = self(0.,
                                  original_encoder_word_input, original_encoder_character_input,
                                  paraphrse_encoder_word_input, paraphrse_encoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)

            cross_entropy = F.cross_entropy(logits, target)

            return cross_entropy, kld

        return validate

    def sample(self, batch_loader, seq_len, seed, use_cuda):
        seed = Variable(torch.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(torch.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        result = ''

        initial_state = None

        for i in range(seq_len):
            logits, initial_state, _ = self(0., None, None,
                                            decoder_word_input, decoder_character_input,
                                            seed, initial_state)

            logits = logits.view(-1, self.params.word_vocab_size)
            prediction = F.softmax(logits)

            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.end_token:
                break

            result += ' ' + word

            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

            decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long())
            decoder_character_input = Variable(torch.from_numpy(decoder_character_input_np).long())

            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        return result
