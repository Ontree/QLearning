#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Merge, Embedding, Reshape, Permute, Reshape)
from keras.models import Model, Sequential
from keras.optimizers import Adam

#import deeprl_hw2 as tfrl
import gym
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import *
from keras import backend as K




def create_model(window, input_shape, num_actions, is_linear,
                 model_name='q_network'):  # noqa: D103
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # the network for processing image
    im_model = Sequential()
    if not is_linear:
        im_model.add(Conv2D(16, (8, 8), activation='relu', input_shape = (input_shape[0], input_shape[1], window), strides = (4, 4)))
        im_model.add(Conv2D(32, (4, 4), activation='relu', strides = (2, 2)))
        im_model.add(Flatten())
        im_model.add(Dense(256, activation='relu'))
        # more hidden layers?
        im_model.add(Dense(num_actions))
    else:
        im_model.add(Reshape((input_shape[0]*input_shape[1]*window,), input_shape=(input_shape[0], input_shape[1], window)))
        im_model.add(Dense(num_actions))

    # mask the action for gradient passing
    # input_dim = num_actions + 1, the embedding of the last one is [1,..1] which is used to get all q values
    embedding = np.identity(num_actions)
    all_actions = np.ones([1, num_actions])
    embedding = np.append(embedding, all_actions, axis = 0)
    action_mask_model = Sequential()
    action_mask_model.add(Embedding(num_actions + 1, num_actions, input_length=
                                   1, weights=[embedding], trainable=False))
    action_mask_model.add(Reshape((num_actions,)))
    model = Sequential()
    model.add(Merge([im_model, action_mask_model], mode='mul'))
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
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    #parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    
    #args.input_shape = tuple(args.input_shape)

    args.output = get_output_folder(args.output, args.env)
    env = gym.make(args.env)
    #env = wrappers.Monitor(env, args.output)
    env.seed(args.seed)

    is_linear = True
    agent = DQNAgent(q_network = create_model(4, (84, 84), env.action_space.n, is_linear),
        preprocessor = AtariPreprocessor((84, 84)),
        memory = None,
        gamma =0.99,
        target_update_freq = 10000,
        num_burn_in = None,
        train_freq = None,
        batch_size = 32,
        is_linear = True,
        model_type = 'double',
        use_replay_and_target_fixing = False,
        epsilon = 0.05)

    agent.compile(lr = 0.0001)
    agent.fit(env, 100)
    agent.load_weights()
    agent.evaluate(env, 10, max_episode_length=None)
    env.close()
    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

if __name__ == '__main__':
    main()
