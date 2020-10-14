import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
tfv1.disable_eager_execution()

from kaggle_environments import make

from q1_schedule import LinearExploration, LinearSchedule
from q3_nature import NatureQN

from configs.connectx import config


class ConnectXEnv(gym.Env):
    """
    To interact with the environment defined by kaggle-environments
    """
    def __init__(self, config):
        self.env = make(config.env_name)
        # most recent raw observations (for max pooling across time steps)
        self.kaggle_config = self.env.configuration
        self.action_space = gym.spaces.Discrete(self.kaggle_config['columns'])
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(self.kaggle_config['rows'], self.kaggle_config['columns'], 1), dtype=np.uint8)
        self.trainer = self.env.train([None, 'negamax'])  # Assumes we go first

    def step(self, action):
        observation, reward, done, info = self.trainer.step(action)
        # Reshape the observation into a 6 x 7 x 1 image
        new_observation = np.array(observation['board']).reshape(self.kaggle_config['rows'], self.kaggle_config['columns'], 1)
        # Reward is None when trying to place in an already-full column: set to -1000 instead
        new_reward = reward if reward is not None else -2

        return new_observation, new_reward, done, info

    def reset(self):
        observation = self.env.reset()[0]['observation']
        new_observation = np.array(observation['board']).reshape(self.kaggle_config['rows'], self.kaggle_config['columns'], 1)
        return new_observation

    def get_available_actions(self, state: np.ndarray):
        assert state.shape == (self.kaggle_config['rows'], self.kaggle_config['columns'], 1)
        return np.where(state[0, :, 0] == 0)[0]  # Check the top row only for available spaces


class ConnectXQN(NatureQN):

    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return np.random.choice(self.env.get_available_actions(state=state))
        else:
            return self.get_best_action(state)[0]

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        with tfv1.variable_scope(scope):
            conv = tfv1.layers.conv2d(state, filters=8, kernel_size=3, strides=1, activation='relu', reuse=reuse, name='conv1')
            #conv2 = tfv1.layers.conv2d(conv1, filters=64, kernel_size=4, strides=2, activation='relu', reuse=reuse, name='conv2')
            #conv3 = tfv1.layers.conv2d(conv2, filters=64, kernel_size=3, strides=1, activation='relu', reuse=reuse, name='conv3')
            dense1 = tfv1.layers.dense(tfv1.layers.flatten(conv), units=32, activation='relu', reuse=reuse, name='dense1')
            out = tfv1.layers.dense(dense1, units=num_actions, reuse=reuse, name='output_layer')

        return out

    def record(self):
        """
        Re create an env and record a video for one episode
        """
        env = ConnectXEnv(config=config)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        self.evaluate(env, 1)


if __name__ == '__main__':
    # make env
    env = ConnectXEnv(config=config)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = ConnectXQN(env, config)
    model.run(exp_schedule, lr_schedule)
