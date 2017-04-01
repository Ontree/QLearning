"""Main DQN agent."""
from keras.optimizers import Adam
import gym
from policy import *
from preprocessors import *
from objectives import *
from core import *
import numpy as np
import utils

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                q_network,
                q_network2,
                preprocessor,
                memory,
                gamma,
                target_update_freq,
                num_burn_in,
                train_freq,
                batch_size,
                is_linear,
                model_type,
                use_replay_and_target_fixing,
                epsilon,
                action_interval,
                save_freq,
                output_path):
        self.q_network = q_network
        self.q_network2 = q_network2
        self.preprocessor = preprocessor 
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.model_type = model_type
        self.use_replay_and_target_fixing = use_replay_and_target_fixing
        self.model_name = ('linear_' if is_linear else 'deep_') + model_type + ('_simple' if use_replay_and_target_fixing else '')
        self.weight_file_name = output_path + '/' + self.model_name + '.h5'
        self.epsilon = epsilon
        self.his_preprocessor = HistoryPreprocessor()
        self.action_interval = action_interval
        self.save_freq = save_freq
        self.output_path = output_path

        

    def compile(self, lr = 0.0001, optimizer_name='adam', loss_func=mean_huber_loss):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        if optimizer_name == 'adam':
            optimizer = Adam(lr=lr)

        self.q_network.compile(optimizer=optimizer,
              loss=loss_func,
              metrics=['mse'] )
        self.q_network2.compile(optimizer=optimizer,
                               loss=loss_func,
                               metrics=['mse'] )
    #def calc_q_values(self, state):
    def calc_q_values(self, state, network):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        Qs = network.predict_on_batch([np.array([state]), np.array([self.n_action])])[0]
        return Qs


    def select_action(self, state, network, policy): #, stage, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        Qs = self.calc_q_values(state, network)
        action = policy.select_action(Qs)
        return action

            

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        samples = self.memory.sample(self.batch_size)
        states = []
        actions = []
        ys = []

        rewards = []
        next_states = []
        terminal_mask = []

        for [state, action, r, next_state, is_terminal] in samples:
            rewards.append(r)
            if is_terminal:
                next_state = np.zeros(state.shape)
            next_states.append(next_state)
            terminal_mask.append(1-is_terminal)
            states.append(state)
            actions.append(action)

        action_mask = self.n_action * np.ones(self.batch_size)
        next_q_max = np.max(self.q_network2.predict_on_batch([np.array(next_states),action_mask ]), axis=1).flatten()
        ys = np.array(rewards) + self.gamma * np.array(terminal_mask) * next_q_max
        states = np.array(states)
        actions = np.array(actions)
        self.q_network.train_on_batch([states, actions], ys)

'''

        for [state, action, r, next_state, is_terminal] in samples:
            if is_terminal:
                y = r
            else:
                y = r + self.gamma * max(self.calc_q_values(next_state, self.q_network2))
            y = [y] * self.n_action

            ys.append(y)
            states.append(state)
            actions.append(action)
        ys = np.array(ys)
        states = np.array(states)
        actions = np.array(actions)
        self.q_network.train_on_batch([states, actions], ys)
'''


       


    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        ses = tf.get_default_session()
        writer = tf.summary.FileWriter(self.output_path, ses)
        writer.add_graph(tf.get_default_graph())

        self.policy = LinearDecayGreedyEpsilonPolicy()
        n_action = env.action_space.n
        self.n_action = n_action
        it = 0
        if self.use_replay_and_target_fixing == False:
            state = env.reset()
            state = self.preprocessor.process_state_for_network(state)
            his_state = self.his_preprocessor.process_state_for_network(state)
            while True:
                it += 1
                action = self.select_action(his_state, self.q_network,
                                            self.policy)
                next_s, r, done, info = env.step(action)
                r = self.preprocessor.process_reward(r)
                if done:
                    y = r
                    self.q_network.fit([np.array([his_state]), np.array([action])],np.array([[y]*n_action]),nb_epoch= 1)
                    state = env.reset()
                    self.his_preprocessor.reset()
                    state = self.preprocessor.process_state_for_network(state)
                    his_state = self.his_preprocessor.process_state_for_network(state)
                    if it >= num_iterations:
                        self.q_network.save_weights(self.weight_file_name)
                        break
                else:
                    old_his = his_state
                    state = next_s
                    state = self.preprocessor.process_state_for_network(state)
                    his_state = self.his_preprocessor.process_state_for_network(state)
                    y = r + self.gamma * max(self.calc_q_values(his_state, self.q_network))
                    self.q_network.fit([np.array([old_his]),
                                        np.array([action])], np.array([[y]*n_action]),nb_epoch = 1)
        else:
            it += self.burn_samples(env)
            epi_num = 0
            while it < num_iterations:
                epi_num += 1
                epi_reward = 0
                state = env.reset()
                self.his_preprocessor.reset()
                #action_countdown = 0
                while True: # start an episode
                    state = self.preprocessor.process_state_for_network(state)
                    his_state = self.his_preprocessor.process_state_for_network(state) 
                    '''
                    if action_countdown == 0: # change action every self.action_interval steps
                        action = self.select_action(his_state, self.q_network, self.policy)
                        action_countdown = self.action_interval
                    '''
                    action = self.select_action(his_state, self.q_network,
                                                self.policy)
                    next_state, reward, is_terminal, info = env.step(action)
                    epi_reward += reward
                    reward = self.preprocessor.process_reward(reward)
                    it += 1
                    #if it%1000 == 0:
                    #    print 'it: ', it
                    #action_countdown -= 1
                    self.memory.append(state, action, reward, is_terminal)
                    state = next_state
                    if it % self.train_freq == 0:
                        self.update_policy()
                    if it % self.target_update_freq == 0:
                        utils.get_hard_target_model_updates(self.q_network2, self.q_network)
                    if it % self.save_freq == 0:
                        self.q_network.save_weights(self.weight_file_name)
                    if is_terminal:
                        utils.add_summary(epi_num, 'reward', epi_reward, writer)
                        if epi_num % 500 == 0:
                            print 'epi: ', epi_num, '  it: ', it
                            evaluate_reward = self.evaluate(env, 20)
                            utils.add_summary(epi_num, 'evaluate_reward', evaluate_reward, writer)
                        break
                    

            self.q_network.save_weights(self.weight_file_name)
    

    # collect samples before starting training
    def burn_samples(self, env):
        print '# collecting samples'
        it = 0
        while  it < self.num_burn_in:
            state = env.reset()
            while  True:
              action = env.action_space.sample()
              next_state, reward, is_terminal, info = env.step(action)
              state = self.preprocessor.process_state_for_network(state)
              reward = self.preprocessor.process_reward(reward)
              self.memory.append(state, action, reward, is_terminal)
              state = next_state
              it += 1
              if it % 100 == 0:
                  print it
              if is_terminal:
                  break
        return it

           
           


    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        policy = GreedyEpsilonPolicy(self.epsilon)
        self.n_action = env.action_space.n
        rewards = []
        for epi in range(num_episodes):
            self.his_preprocessor.reset();
            state = env.reset();
            reward = 0
            while True: 
              state = self.preprocessor.process_state_for_network(state)
              his_state = self.his_preprocessor.process_state_for_network(state)
              action = self.select_action(his_state, self.q_network,policy)
              state, r, done, info = env.step(action)
              reward += r
              if done:
                  rewards.append(reward)
                  print epi, reward
                  break
        print 'average reward: ', np.mean(rewards)
        return np.mean(rewards)





    def load_weights(self):
        self.q_network.load_weights(self.weight_file_name)


