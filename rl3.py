import gym
import sys
from lib.envs.cliff_walking import CliffWalkingEnv
import mujoco_py
import os
#env = gym.make('CartPole-v0')
#env=gym.make("Ant-v2")
env=gym.make("Humanoid-v2")
#env = gym.make('BreakoutNoFrameskip-v4')
#env=CliffWalkingEnv()
#print(env.observation_space.n)
'''
for i_episode in range(1):
    observation = env.reset()
    print(observation)
    env.render()
    #print(observation.shape)
    #print(observation)
    for step in range(10000):
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print(reward)
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break
'''

import numpy as np
a=np.array([[3,2],[2,3]])
print(a)
b=a.reshape([1,4])
print(b)