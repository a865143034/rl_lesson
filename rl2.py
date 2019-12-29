#coding:utf-8
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from _policies import BinaryActionLinearPolicy


env = gym.make("CartPole-v0")
n_iter=10
batch_size=25
elite_frac = 0.2
th_mean = np.zeros(env.observation_space.shape[0]+1)
initial_std = 1.0
n_elite = int(np.round(batch_size*elite_frac))
th_std = np.ones_like(th_mean) * initial_std
num_steps=1000

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1


def noisy_evaluation(theta):
    agent = BinaryActionLinearPolicy(theta) # 根据权重重新制定策略
    rew, T = do_rollout(agent, env, num_steps) # 计算reward
    return rew


for x in range(n_iter):
    # 更新权重
    ths = np.array([th_mean + dth for dth in th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
    ys = np.array([noisy_evaluation(th) for th in ths]) # 计算reward
    elite_inds = ys.argsort()[::-1][:n_elite]
    elite_ths = ths[elite_inds] # 把ths按reward排序
    th_mean = elite_ths.mean(axis=0)
    th_std = elite_ths.std(axis=0)