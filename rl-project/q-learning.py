import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from collections import defaultdict
from lib import plotting
#from lib.envs.cliff_walking import CliffWalkingEnv
import cv2
env_name = 'BreakoutNoFrameskip-v4'
env=gym.make(env_name)


def process_img(obs):
    #print(obs.shape)
    str1=""
    for i in obs:
        for j in i:
            for k in j:
                str1+=str(k)
                str1+="_"
    #print(str1)
    return str1


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(obs):
        A = np.ones(nA, dtype=float) * epsilon / nA
        #print(obs)
        best_action = np.argmax(Q[obs])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        #debug
        #print(i_episode)
        state = env.reset()
        state=process_img(state)

        for t in itertools.count():
            action_probs = policy(state)
            #print(action_probs)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            #env.render()

            next_state, reward, done, _ = env.step(action)
            next_state=process_img(next_state)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            print("\rStep {} @ Episode {}/{}".format(t, i_episode + 1, num_episodes), end="")
            if done:
                break
            state = next_state
    return Q, stats

Q, stats = q_learning(env, 100)

plotting.plot_episode_stats(stats)