#coding:utf-8
import gym
import numpy as np
from matplotlib import pyplot as plt

env_name='Breakout-v0'
#env_name='CartPole-v0'
env=gym.make(env_name)
obs=env.reset()
print(obs.shape)

def random_policy(n):
    action=np.random.randint(0,n)
    return action

'''
for step in range(10000):
    action = np.random.randint(0,env.action_space.n)
    obs,reward,done,info=env.step(action)
    env.render()

    if done:
        img=env.render(mode='rgb-array')
        #plt.imshow(img)
        #plt.show()
        print("the game is over in {} steps".format(step))
        break
env.close()
'''
import numpy as np

# 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
a1 = np.random.choice(a=5, size=1, replace=False, p=None)
print(a1)
# 非一致的分布，会以多少的概率提出来
a2 = np.random.choice(a=5, size=1, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
print(a2)


import collections

# 两种方法来给 namedtuple 定义方法名
#User = collections.namedtuple('User', ['name', 'age', 'id'])
User = collections.namedtuple('User', 'name age id')
user = User('tester', '22', '464643123')

print(user)

import matplotlib.pyplot as plt
fig1=plt.figure()
plt.plot([1.,2.,3.,4.])
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Episode Length over Time")
plt.show(fig1)