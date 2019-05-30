import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from torch.autograd import Variable
from model.rvae import RVAE

if __name__ == '__main__':

    assert os.path.exists('trained_RVAE'), \
        'trained model not found'

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    # parser.add_argument('--num-sample', type=int, default=10, metavar='NS',
    #                     help='num samplings (default: 10)')

    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    rvae.load_state_dict(torch.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    seq_len = 50
    seed = np.random.normal(size=[1, parameters.latent_variable_size])

    data = [["how are you ?"], ["how are you doing"]]
    data_words = [[line.split() for line in target] for target in data]
    word_tensor = np.array([[list(map(batch_loader.word_to_idx.get, line)) for line in target] for target in data_words])

    character_tensor = np.array(
        [[list(map(batch_loader.encode_characters, line)) for line in target] for target in data_words])

    original_encoder_word_input = [seq_tensor for seq_tensor in word_tensor[0]]
    original_encoder_character_input = [character_tensor_seq for character_tensor_seq in character_tensor[0]]
    input_seq_len = [len(line) for line in original_encoder_word_input]
    ref_max_input_seq_len = np.amax(input_seq_len)

    paraphrse_encoder_word_input = [seq_tensor for seq_tensor in word_tensor[1]]
    paraphrse_encoder_character_input = [character_tensor_seq for character_tensor_seq in character_tensor[1]]
    para_input_seq_len = [len(line) for line in paraphrse_encoder_word_input]
    para_max_input_seq_len = np.amax(para_input_seq_len)

    max_input_seq_len = max(ref_max_input_seq_len, para_max_input_seq_len)

    for i, line in enumerate(original_encoder_word_input):
        line_len = input_seq_len[i]
        to_add = max_input_seq_len - line_len
        original_encoder_word_input[i] = [batch_loader.word_to_idx[batch_loader.pad_token]] * to_add + line[::-1]

    for i, line in enumerate(original_encoder_character_input):
        line_len = input_seq_len[i]
        to_add = max_input_seq_len - line_len
        original_encoder_character_input[i] = [batch_loader.encode_characters(batch_loader.pad_token)] * to_add + line[::-1]

    for i, line in enumerate(paraphrse_encoder_word_input):
        line_len = para_input_seq_len[i]
        to_add = max_input_seq_len - line_len
        paraphrse_encoder_word_input[i] = [batch_loader.word_to_idx[batch_loader.pad_token]] * to_add + line[::-1]

    for i, line in enumerate(paraphrse_encoder_character_input):
        line_len = para_input_seq_len[i]
        to_add = max_input_seq_len - line_len
        paraphrse_encoder_character_input[i] = [batch_loader.encode_characters(batch_loader.pad_token)] * to_add + line[::-1]

    input = [original_encoder_word_input, original_encoder_character_input, paraphrse_encoder_word_input,
             paraphrse_encoder_character_input]

    if args.use_cuda:
        seed = seed.cuda()
    input = [Variable(torch.from_numpy(var)) for var in input]
    input = [var.long() for var in input]
    input = [var.cuda() if use_cuda else var for var in input]

    [original_encoder_word_input, original_encoder_character_input, paraphrse_encoder_word_input,
     paraphrse_encoder_character_input] = input
    decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

    decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long())
    decoder_character_input = Variable(torch.from_numpy(decoder_character_input_np).long())

    if args.use_cuda:
        decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

    result = ''

    initial_state = None
    sentence = " ".join(
        list(map(lambda x: batch_loader.idx_to_word[x], original_encoder_word_input[0].tolist()))[::-1])
    reference = " ".join(
        list(map(lambda x: batch_loader.idx_to_word[x], paraphrse_encoder_word_input[0].tolist()))[::-1])
    for i in range(seq_len):
        logits, initial_state, _ = rvae.forward(0., original_encoder_word_input[0:1],
                                                original_encoder_character_input[0:1],
                                                paraphrse_encoder_word_input[0:1],
                                                paraphrse_encoder_character_input[0:1],
                                                decoder_word_input[0:1], decoder_character_input[0:1],
                                                z=seed, initial_state=initial_state)

        logits = logits.view(-1, rvae.params.word_vocab_size)
        prediction = F.softmax(logits)

        word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

        if word == batch_loader.end_token:
            break

        result += ' ' + word

        decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
        decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

        decoder_word_input = Variable(torch.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(torch.from_numpy(decoder_character_input_np).long())

        if args.use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

    print(sentence, reference, result)