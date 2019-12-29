import gym
from lib.envs.cliff_walking import CliffWalkingEnv
def action(status):
    pos,v,ang,va=status
    #print(status)
    if ang>0:return 1
    else:return 0

env = gym.make('BreakoutNoFrameskip-v4')
env = CliffWalkingEnv()
print(env.observation_space.sample())
observation=env.reset()
for _ in range(1):
    env.render()
    observation, reward, done, info=env.step(action(observation)) # take a random action
    #print(observation.shape)
    if done:
        print("dead in %d steps" %_)
        break;
env.close()