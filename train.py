import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE


def create_batches(data):
    num_samples = len(data[0])
    permutation_ = permutation(num_samples)

    [original_encoder_word_input, original_encoder_character_input, paraphrse_encoder_word_input,
     paraphrse_encoder_character_input, decoder_word_input, decoder_character_input, target] = data[permutation_]

    original_encoder_word_input_batches = [original_encoder_word_input[x: x + args.batch_size]
                                           for x in range(0, len(original_encoder_word_input), args.batch_size)]

    original_encoder_character_input_batches = [original_encoder_character_input[x: x + args.batch_size]
                                                for x in
                                                range(0, len(original_encoder_character_input), args.batch_size)]

    paraphrse_encoder_word_input_batches = [paraphrse_encoder_word_input[x: x + args.batch_size]
                                            for x in range(0, len(paraphrse_encoder_word_input), args.batch_size)]

    paraphrse_encoder_character_input_batches = [paraphrse_encoder_character_input[x: x + args.batch_size]
                                                 for x in
                                                 range(0, len(paraphrse_encoder_character_input), args.batch_size)]

    decoder_word_input_batches = [decoder_word_input[x: x + args.batch_size]
                                  for x in range(0, len(decoder_word_input), args.batch_size)]

    decoder_character_input_batches = [decoder_character_input[x: x + args.batch_size]
                                       for x in range(0, len(decoder_character_input), args.batch_size)]

    target_batches = [target[x: x + args.batch_size]
                      for x in range(0, len(target), args.batch_size)]

    return original_encoder_word_input_batches, original_encoder_character_input_batches, \
           paraphrse_encoder_word_input_batches, paraphrse_encoder_character_input_batches, \
           decoder_word_input_batches, decoder_character_input_batches, target_batches


if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')

    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    if args.use_trained:
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer)
    # validate = rvae.validater()

    ce_result = []
    kld_result = []
    # training_data = batch_loader.training_data('train')
    # validation_data = batch_loader.training_data('valid')

    for iteration in range(args.num_iterations):
        # [original_encoder_word_input_batches, original_encoder_character_input_batches,
        #  paraphrse_encoder_word_input_batches, paraphrse_encoder_character_input_batches,
        #  decoder_word_input_batches, decoder_character_input_batches, target_batches] = create_batches(training_data)
        #
        # [val_original_encoder_word_input_batches, val_original_encoder_character_input_batches,
        #  val_paraphrse_encoder_word_input_batches, val_paraphrse_encoder_character_input_batches,
        #  val_decoder_word_input_batches, val_decoder_character_input_batches, val_target_batches] = create_batches(
        #     validation_data)
        x = 0
        while True:
            input = batch_loader.next_batch(x, args.batch_size, "train")
            if input is None:
                break
            # input = [original_encoder_word_input_batches[batch_no],
            # original_encoder_character_input_batches[batch_no],
            #          paraphrse_encoder_word_input_batches[batch_no],
            #          paraphrse_encoder_character_input_batches[batch_no], decoder_word_input_batches[batch_no],
            #          decoder_character_input_batches[batch_no], target_batches[batch_no]]

            cross_entropy, kld, coef = train_step(iteration, input, args.use_cuda, args.dropout)

            if iteration % 2 == 0:
                print('\n')
                print('------------TRAIN-------------')
                print('----------ITERATION-----------')
                print(iteration)
                print('--------CROSS-ENTROPY---------')
                print(cross_entropy.data.cpu().numpy()[0])
                print('-------------KLD--------------')
                print(kld.data.cpu().numpy()[0])
                print('-----------KLD-coef-----------')
                print(coef)
                print('------------------------------')
            x += args.batch_size

        y = 0
        while True:
            input = batch_loader.next_batch(y, args.batch_size, "valid")
            if input is None:
                break
        # for batch_no in range(len(val_original_encoder_word_input_batches)):
        #     input = [val_original_encoder_word_input_batches[batch_no],
        #              val_original_encoder_character_input_batches[batch_no],
        #              val_paraphrse_encoder_word_input_batches[batch_no],
        #              val_paraphrse_encoder_character_input_batches[batch_no],
        #              val_decoder_word_input_batches[batch_no],
        #              val_decoder_character_input_batches[batch_no], val_target_batches[batch_no]]

            cross_entropy, kld, coef = train_step(iteration, input, args.use_cuda, args.dropout)

            if iteration % 5 == 0:
                print('\n')
                print('------------TRAIN-------------')
                print('----------ITERATION-----------')
                print(iteration)
                print('--------CROSS-ENTROPY---------')
                print(cross_entropy.data.cpu().numpy()[0])
                print('-------------KLD--------------')
                print(kld.data.cpu().numpy()[0])
                print('-----------KLD-coef-----------')
                print(coef)
                print('------------------------------')
            y += args.batch_size

        if iteration % 2 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])
            # input = [val_original_encoder_word_input_batches[batch_no][0:1],
            #          val_original_encoder_character_input_batches[batch_no][0:1],
            #          val_paraphrse_encoder_word_input_batches[batch_no][0:1],
            #          val_paraphrse_encoder_character_input_batches[batch_no][0:1]]
            sample = rvae.predict(input, batch_loader, 50, seed, args.use_cuda)

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print(sample.encode("utf-8"))
            print('------------------------------')
        print("--------------------saving checkpoint-----------------------")
        t.save(rvae.state_dict(), 'trained_RVAE_checkpoint')
        print("--------------------saved checkpoint-----------------------")

    t.save(rvae.state_dict(), 'trained_RVAE')

    np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))
