import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

env = gym.envs.make("Humanoid-v2")
class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [376], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            '''
            def weight_variable(shape):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial)

            def bias_variable(shape):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)

            def conv2d(x, W):
                # stride[1, x_movement, y_movement, 1]
                # Must have strides[0] = strides[3] =1
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")  # padding="SAME"用零填充边界

            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            '''
            self.state1=tf.reshape(self.state,(47,8,1))
            #print(self.state.shape)
            #print(self.state)
            self.out1=tf.contrib.layers.conv2d(
                inputs=tf.expand_dims(self.state1,0),
                num_outputs=32,
                kernel_size=[3,3],
                stride=1,
            )#[1-,47,8,1]
            #print(self.out1.shape)
            self.pool1 = tf.contrib.layers.max_pool2d(inputs=self.out1, kernel_size=2, stride=2, padding='SAME') #[-1,12,12,64]

            print(self.pool1.shape)

            self.out2=tf.contrib.layers.conv2d(
                inputs=self.pool1,
                num_outputs=32,
                kernel_size=[3,3],
                stride=2,
            )
            print(self.out2.shape)
            self.pool2 = tf.contrib.layers.max_pool2d(inputs=self.out2, kernel_size=2, stride=2, padding='SAME')
            print(self.pool2.shape)
            self.pool2=tf.reshape(self.pool2,[-1])
            print(self.pool2.shape)
            # This is just linear classifier
            self.tmp1 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.pool2,0),
                num_outputs=1,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.zeros_initializer)
            print(self.tmp1.shape)

            self.mu = tf.contrib.layers.fully_connected(
                inputs=self.tmp1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.mu = tf.squeeze(self.mu)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.pool2,0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            print(self.mu.shape)
            #print(self.sigma.shape)

            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
            #print(self.action)

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [376], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.state1=tf.reshape(self.state,(47,8,1))
            #print(self.state.shape)
            #print(self.state)
            self.out1=tf.contrib.layers.conv2d(
                inputs=tf.expand_dims(self.state1,0),
                num_outputs=32,
                kernel_size=[3,3],
                stride=1,
            )#[1-,47,8,1]
            #print(self.out1.shape)
            self.pool1 = tf.contrib.layers.max_pool2d(inputs=self.out1, kernel_size=2, stride=2, padding='SAME') #[-1,12,12,64]

            #print(self.pool1.shape)

            self.out2=tf.contrib.layers.conv2d(
                inputs=self.pool1,
                num_outputs=32,
                kernel_size=[3,3],
                stride=2,
            )
            #print(self.out2.shape)
            self.pool2 = tf.contrib.layers.max_pool2d(inputs=self.out2, kernel_size=2, stride=2, padding='SAME')

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.pool2,
                num_outputs=1,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()

        episode = []

        # One step in the environment
        for t in itertools.count():

            # env.render()

            # Take a step
            action = estimator_policy.predict(state)
            #print(action.shape)
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            estimator_value.update(state, td_target)
            estimator_policy.update(state, td_error, action)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if done:
                break

            state = next_state

    return stats


tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(learning_rate=0.001)
value_estimator = ValueEstimator(learning_rate=0.1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = actor_critic(env, policy_estimator, value_estimator,2000, discount_factor=0.95)


plotting.plot_episode_stats(stats, smoothing_window=10)
