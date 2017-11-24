"""
Lasagne implementation of SGDR on WRNs from "SGDR: Stochastic Gradient Descent with Restarts"
(http://arxiv.org/abs/XXXX) This code is based on Lasagne Recipes available at
https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
and on WRNs implementation by Florian Muellerklein available at
https://gist.github.com/FlorianMuellerklein/3d9ba175038a3f2e7de3794fa303f1ee

"""

from __future__ import print_function

import sys
import os
import time
import pickle
from argparse import ArgumentParser

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, batch_norm, BatchNormLayer
from lasagne.layers import ElemwiseSumLayer, NonlinearityLayer, GlobalPoolLayer
from lasagne.init import HeNormal
from lasagne.layers import Conv2DLayer as ConvLayer

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

num_of_train_images = 1281167

class Logger:
    def __init__(self, k, lr, run):
        self.lr = lr
        self.k = k
        self.run = run

    def log_message(self, message):
        with open('log_{}_{}_{}.txt'.format(self.k, self.lr, self.run), 'a') as l_f:
            l_f.write(message + '\n')
            l_f.flush()

    def log_stat(self, message):
        with open("stat_{}_{}_{}.txt".format(self.k, self.lr, self.run), 'a') as l_f:
            l_f.write(message)
            l_f.flush()

    def log_loss(self, message):
        with open("statloss_{}_{}_{}.txt".format(self.k, self.lr, self.run), 'a') as l_f:
            l_f.write(message)
            l_f.flush()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# Mean image can be extracted from any training data file
def load_validation_data(data_folder, mean_image, img_size=32):
    test_file = os.path.join(data_folder, 'val_data')

    d = unpickle(test_file)
    x = d['data']
    y = d['labels']
    x = x / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = np.array([i-1 for i in y])

    # Remove mean (computed from training data) from images
    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return dict(
        X_test=lasagne.utils.floatX(x),
        Y_test=y.astype('int32'))


def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        mean=mean_image)


# ##################### Build the neural network model #######################


def ResNet_FullPre_Wide(input_var=None, nout=10, n=3, k=2, dropoutrate=0, img_size=32):
    '''
    Adapted from https://gist.github.com/FlorianMuellerklein/3d9ba175038a3f2e7de3794fa303f1ee
    which was tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016
    (https://arxiv.org/abs/1603.05027)
    And 'Wide Residual Networks', Sergey Zagoruyko, Nikos Komodakis 2016 (http://arxiv.org/pdf/1605.07146v1.pdf)
    '''
    n_filters = {0: 16, 1: int(16*k), 2: int(32*k), 3: int(64*k), 4: int(128*k)}

    # create a residual learning building block with two stacked 3x3 convlayers and dropout
    def residual_block(l, increase_dim=False, first=False, filters=16):
        if increase_dim:
            first_stride = (2, 2)
        else:
            first_stride = (1, 1)

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(3,3), stride=first_stride,
                                      nonlinearity=rectify, pad='same', W=HeNormal(gain='relu')))

        if dropoutrate > 0:   # with dropout
            dropout = DropoutLayer(conv_1, p=dropoutrate)

            # contains the last weight portion, step 6
            conv_2 = ConvLayer(dropout, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None,
                               pad='same', W=HeNormal(gain='relu'))
        else:   # without dropout
            conv_2 = ConvLayer(conv_1, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None,
                               pad='same', W=HeNormal(gain='relu'))

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None,
                                   pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])
        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None,
                                   pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])
        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, img_size, img_size), input_var=input_var)

    # first layer
    l = batch_norm(ConvLayer(l_in, num_filters=n_filters[0], filter_size=(3,3), stride=(1,1), nonlinearity=rectify,
                             pad='same', W=HeNormal(gain='relu')))

    # first stack of residual blocks
    l = residual_block(l, first=True, filters=n_filters[1])
    for _ in range(1, n):
        l = residual_block(l, filters=n_filters[1])

    # second stack of residual blocks
    l = residual_block(l, increase_dim=True, filters=n_filters[2])
    for _ in range(1, n):
        l = residual_block(l, filters=n_filters[2])

    # third stack of residual blocks
    if img_size >= 32:
        l = residual_block(l, increase_dim=True, filters=n_filters[3])
        for _ in range(1, n):
            l = residual_block(l, filters=n_filters[3])

    # fourth stack of residual blocks
    if img_size >= 64:
        l = residual_block(l, increase_dim=True, filters=n_filters[4])
        for _ in range(1, n):
            l = residual_block(l, filters=n_filters[4])

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # fully connected layer
    network = DenseLayer(avg_pool, num_units=nout, W=HeNormal(), nonlinearity=softmax)

    return network

# ############################# Batch iterator ###############################


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False, img_size=32):

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = \
                    padded[r, :, crops[r, 0]:(crops[r, 0]+img_size), crops[r, 1]:(crops[r, 1]+img_size)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


# ############################## Main program ################################

def main(data_folder, n=4, irun=1, k=1, num_epochs=40, cont=False, E1=10, E2=20, E3=30, lr=0.1, lr_fac=0.1,
         reg_fac=0.0005, dropoutrate=0, img_size=32):

    nout = 1000
    logger = Logger(k, lr, irun)

    # Load the dataset
    logger.log_message("Loading data...")

    # Load first batch so we can extract mean image needed to load validation data
    data = load_databatch(data_folder, 1, img_size=img_size)
    mean_image = data['mean']
    del data

    # Load test data
    test_data = load_validation_data(data_folder, mean_image=mean_image, img_size=img_size)
    X_test = test_data['X_test']
    Y_test = test_data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    logger.log_message("Building model and compiling functions...")
    network = ResNet_FullPre_Wide(input_var, nout,  n, k, dropoutrate, img_size)
    logger.log_message("Number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    print("Number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    print('Img Size %d' % img_size)
    print('K %d' % k)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # Add weight decay
    all_layers = lasagne.layers.get_all_layers(network)
    sh_reg_fac = theano.shared(lasagne.utils.floatX(reg_fac))
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * sh_reg_fac
    loss = loss + l2_penalty

    # Create update expressions for training
    # Stochastic Gradient Descent (SGD) with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

    test_loss = test_loss.mean()
    test_acc_1 = T.mean(lasagne.objectives.categorical_accuracy(test_prediction, target_var),
                        dtype=theano.config.floatX)
    test_acc_5 = T.mean(lasagne.objectives.categorical_accuracy(test_prediction, target_var, 5),
                        dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc_1, test_acc_5])

    start_time0 = time.time()

    batchsize = 128
    start_epoch = 0
    # Load model #####################################################################
    if cont:
        filename = 'network_last_{}_{}.p'.format(lr, run)
        logger.log_message('Loading network from file %s' % filename)
        net = unpickle(filename)
        start_epoch = net['epoch']
        for p, value in zip(updates.keys(), net['u']):
            p.set_value(value)
        lasagne.layers.set_all_param_values(network, net['w'], trainable=False)

    # Simulate learning rate runs
    for epoch in range(start_epoch):
        # Adjust learning rate
        if (epoch + 1) == E1 or (epoch + 1) == E2 or (epoch + 1) == E3:
            new_lr = sh_lr.get_value() * lr_fac
            logger.log_message("New LR:" + str(new_lr))
            sh_lr.set_value(lasagne.utils.floatX(new_lr))

    # Training #####################################################################
    logger.log_message("Starting training...")

    # We iterate over epochs:
    for epoch in range(start_epoch, num_epochs):
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()

        for idatabatch in range(1, 11):
            start_time_tmp = time.time()
            data = load_databatch(data_folder, idatabatch, img_size=img_size)
            print('Data loading took %f' % (time.time() - start_time_tmp))
            X_train = data['X_train']
            Y_train = data['Y_train']

            train_err = 0
            train_batches = 0

            for batch in iterate_minibatches(X_train, Y_train, batchsize, shuffle=True, augment=True, img_size=img_size):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            logger.log_loss("{}\t{:.15g}\t{:.15g}\t{:.15g}\n".format(epoch, float(sh_lr.get_value()),
                time.time() - start_time0, train_err / train_batches))

            logger.log_message("idatabatch#{} took {:.3f}s".format(idatabatch, time.time() - start_time))
            del data, X_train, Y_train

        print('Train Data pass took: %f' % (time.time() - start_time))
        # And a full pass over the validation data:
        val_err = 0
        val_acc_1 = 0
        val_acc_5 = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False, img_size=img_size):
            inputs, targets = batch
            err, acc_1, acc_5 = val_fn(inputs, targets)
            val_err += err
            val_acc_1 += acc_1
            val_acc_5 += acc_5
            val_batches += 1

        print('Epoch took: %f' % (time.time() - start_time))
        # Then we print the results for this epoch:
        logger.log_message("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        logger.log_message("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        logger.log_message("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        logger.log_message("  validation accuracy_1:\t\t{:.2f} %".format(val_acc_1 / val_batches * 100))
        logger.log_message("  validation accuracy_5:\t\t{:.2f} %".format(val_acc_5 / val_batches * 100))

        # Print some statistics
        logger.log_stat("{}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\n"
                        .format(epoch, float(sh_lr.get_value()), time.time() - start_time0,
                                train_err / train_batches, val_err / val_batches,
                                val_acc_1 / val_batches * 100, val_acc_5 / val_batches * 100))

        # Get network parameters and save it
        net = {
            'u': [p.get_value() for p in updates.keys()],
            'w': lasagne.layers.get_all_param_values(network, trainable=False),
            'epoch': (epoch+1)
        }

        # pickle.dump(net, open("network_{}_{}_{}.p".format(lr, irun, epoch+1), 'wb'))
        pickle.dump(net, open("network_last_{}_{}.p".format(lr, irun), 'wb'))

        # Adjust learning rate
        if (epoch+1) == E1 or (epoch+1) == E2 or (epoch+1) == E3:
            new_lr = sh_lr.get_value() * lr_fac
            logger.log_message("New LR:"+str(new_lr))
            sh_lr.set_value(lasagne.utils.floatX(new_lr))

    # Calculate validation error of model:
    test_err = 0
    test_acc_1 = 0
    test_acc_5 = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc_1, acc_5 = val_fn(inputs, targets)
        test_err += err
        test_acc_1 += acc_1
        test_acc_5 += acc_5
        test_batches += 1
    logger.log_message("Final results:")
    logger.log_message("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    logger.log_message("  test accuracy 1:\t\t{:.2f} %".format(test_acc_1 / test_batches * 100))
    logger.log_message("  test accuracy 5:\t\t{:.2f} %".format(test_acc_5 / test_batches * 100))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-s', '--img_size', help="Size of images, represented as string '32x32' or '64x64'",
                        default=32, type=int)
    parser.add_argument('-lr', '--learning_rate', help="Starting Learning Rate, "
                                                       "decreased by the factor of 5 every 10 epochs",
                        default=0.01, type=float)
    parser.add_argument('-k', '--network_width', help="Network width hyper-parameter. Number of filters in each layer "
                                                      "is multiplied by this factor", default=1, type=float)
    parser.add_argument('-r', '--run', help="Number used to index output files, helpful when multiple runs required",
                        default=1, type=int)
    parser.add_argument('-c', '--cont', help="Read last saved model and continue training from that point",
                        default=False, type=bool)
    parser.add_argument('-df', '--data_folder', help="Path to the folder containing training and validation data",
                        required=True)
    parser.add_argument('-d', '--decay', help="L2 decay", default=0.0005, type=float)
    args = parser.parse_args()

    return args.img_size, args.learning_rate, args.network_width, args.run, args.cont, args.data_folder, args.decay


if __name__ == '__main__':
    img_size, lr, k, run, cont, data_folder, reg_fac = parse_arguments()

    lr_fac = 0.2
    num_epochs = 40
    E1 = 10
    E2 = 20
    E3 = 30
    Estart = 10000
    n = 4
    dropout = 0


    main(data_folder, n, run, k, num_epochs, cont, E1, E2, E3, lr, lr_fac, reg_fac, dropout, img_size)