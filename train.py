import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE
import progressbar


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
        print(f"-----Iteration: {iteration}-------------")
        x = 0
        bar = progressbar.ProgressBar(maxval=130001,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        while True:
            input = batch_loader.next_batch(x, args.batch_size, "train")
            bar.update(x + 1)
            if input is None:
                break
            cross_entropy, kld, coef = train_step(iteration, input, args.use_cuda, args.dropout)
            x += args.batch_size
            if x == 64:
                seed = np.random.normal(size=[1, parameters.latent_variable_size])
                sentence, reference, result = rvae.predict(input, batch_loader, 50, seed, args.use_cuda)

                print('\n')
                print('------------SAMPLE------------')
                print('------------------------------')
                print(f'original: {sentence.encode("utf-8")}')
                print(f'reference: {reference.encode("utf-8")}')
                print(f'generated: {result.encode("utf-8")}')
                print('------------------------------')
        bar.finish()

        y = 0
        bar = progressbar.ProgressBar(maxval=19262,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        while True:
            input = batch_loader.next_batch(y, args.batch_size, "valid")
            bar.update(y + 1)
            if input is None:
                break
            cross_entropy, kld, coef = train_step(iteration, input, args.use_cuda, args.dropout)
            y += args.batch_size
            if y == 64:
                seed = np.random.normal(size=[1, parameters.latent_variable_size])
                sentence, reference, result = rvae.predict(input, batch_loader, 50, seed, args.use_cuda)

                print('\n')
                print('------------SAMPLE------------')
                print('------------------------------')
                print(f'original: {sentence.encode("utf-8")}')
                print(f'reference: {reference.encode("utf-8")}')
                print(f'generated: {result.encode("utf-8")}')
                print('------------------------------')
        bar.finish()

        if iteration % 10 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy.data.cpu().numpy())
            print('-------------KLD--------------')
            print(kld.data.cpu().numpy())
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')

        print("--------------------saving checkpoint-----------------------")
        t.save(rvae.state_dict(), 'trained_RVAE_checkpoint_para_out')
        print("--------------------saved checkpoint-----------------------")
        print("\n\n\n")

    t.save(rvae.state_dict(), 'trained_RVAE')

    np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))
