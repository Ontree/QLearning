#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, convolutional, Dense, Flatten, Input,
                          Merge, Embedding, Reshape, Permute, Reshape, multiply, Lambda)
from keras.models import Model, Sequential
from keras.optimizers import Adam

#import deeprl_hw2 as tfrl
import gym
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import *
from deeprl_hw2.core import *
from keras import backend as K





def create_model(window, input_shape, num_actions, is_linear,
                 model_type):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    # the network for processing image
    assert(is_linear and model_type!='q')
    im_input = Input((input_shape[0], input_shape[1], window))
    if not is_linear:
        '''
        conv_layer1 = convolutional.Conv2D(16, (8, 8), activation='relu', strides = (4,
                                                                       4))(im_input)
        conv_layer2 = convolutional.Conv2D(32, (4, 4), activation='relu', strides = (2,
                                                                       2))(conv_layer1)
        flatten_layer = Flatten()(conv_layer2)
        fc_layer1 = Dense(256, activation='relu')(flatten_layer)
        '''
        conv_layer1 = convolutional.Conv2D(32, (8, 8), activation='relu', strides = (4, 4))(im_input)
        conv_layer2 = convolutional.Conv2D(64, (4, 4), activation='relu', strides = (2, 2))(conv_layer1)
        conv_layer3 = convolutional.Conv2D(64, (3, 3), activation='relu', strides = (1, 1))(conv_layer2)
        flatten_layer = Flatten()(conv_layer3)
        fc_layer1 = Dense(512, activation='relu')(flatten_layer)
        if model_type == 'dueling':
            fc_layer2 = Dense(num_actions + 1, activation='relu')(fc_layer1)
            action_layer = Lambda(lambda x: x[:, 0] + x[:, 1:] - K.mean(x[:, 1:], keepdims = True))(fc_layer2)
        else:
            action_layer = Dense(num_actions)(fc_layer1)
    else:
        reshape_layer = Reshape((input_shape[0]*input_shape[1]*window,))(im_input)
        action_layer = Dense(num_actions)(reshape_layer)
    # mask the action for gradient passing
    # input_dim = num_actions + 1, the embedding of the last one is [1,..1] which is used to get all q values
    embedding = np.identity(num_actions)
    all_actions = np.ones([1, num_actions])
    embedding = np.append(embedding, all_actions, axis = 0)
    action_input = Input(shape=(1,))
    embedding_layer = Embedding(num_actions + 1, num_actions, input_length=
                                   1, weights=[embedding], trainable=False)(action_input)
    mask_layer = Reshape((num_actions,))(embedding_layer)
    output_layer = multiply([action_layer, mask_layer])
    model = Model([im_input, action_input], output_layer)
    return model






def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    #parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--output', default='results', help='Directory to save data to')
    parser.add_argument('-l', '--isLinear', default=0, type=int, choices=range(0, 2), help='1: use linear model; 0: use deep model')
    parser.add_argument('-m', '--modelType', default='q', choices=['q', 'double', 'dueling'], help='q: q learning; double: double q learning; dueling: dueling q learning')
    parser.add_argument('-s', '--simple', default = 0, type=int, choices=range(0, 2), help='1: without replay or target fixing ; 0: use replay and target fixing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    
    #args.input_shape = tuple(args.input_shape)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    model_name = ('linear_' if args.isLinear else 'deep_') + args.modelType + ('_simple' if args.simple else '')
    args.output = get_output_folder(args.output + '/' + model_name, args.env)
    env = gym.make(args.env)
    #env = gym.wrappers.Monitor(env, args.output)
    env.seed(args.seed)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)
    K.get_session().run(tf.initialize_all_variables())
    
    is_linear = args.isLinear
    agent = DQNAgent(q_network = create_model(4, (84, 84), env.action_space.n, is_linear, args.modelType),
        q_network2 = create_model(4, (84, 84), env.action_space.n, is_linear, args.modelType),
        preprocessor = AtariPreprocessor((84, 84)),
        memory = ReplayMemory(1000000, 4),
        gamma =0.99,
        target_update_freq = 10000,
        num_burn_in = 50000,
        train_freq = 4,
        batch_size = 32,
        is_linear = is_linear,
        model_type = args.modelType,
        use_replay_and_target_fixing = (not args.simple),
        epsilon = 0, #0.05,
        action_interval = 4,
        output_path = args.output,
        save_freq = 100000)

    agent.compile(lr = 0.0001)
    agent.fit(env, 5000000)
    agent.load_weights()
    agent.evaluate(env, 100, video_path_suffix='final')
    env.close()
    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
