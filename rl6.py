#coding:utf-8
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])

import tensorflow as tf
print(tf.__version__)
import tensorflow as tf
import numpy as np

a=np.array([1,2,4,5,3])
a=a.reshape([1,6])
print(a)