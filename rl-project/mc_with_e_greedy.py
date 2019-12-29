#coding:utf-8
import gym
import matplotlib
import numpy as np
import sys
import sklearn
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing
from collections import defaultdict
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env_name='BreakoutNoFrameskip-v4'
env = gym.make(env_name)
'''
# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(1000)])
scaler = sklearn.preprocessing.StandardScaler()
observation_examples=observation_examples.reshape([1000,100800])
print(observation_examples.shape)
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))
print(observation_examples.shape)
def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform(np.array(state).reshape([1,100800]))
    featurized = featurizer.transform(scaled)
    #print(featurized.shape)
    return featurized[0]
'''
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
    #nA是动作总数
    def policy_fn(observation):
        #return的是一个list，代表每个动作的概率
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        #Q是动作价值函数
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(1, num_episodes + 1):
        #可以去掉，便于debug
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()
        state=process_img(state)
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            next_state=process_img(next_state)

            #env.render()

            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        #print("break")
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

    return Q, policy

print(env.action_space.n)
#本来500000
Q, policy = mc_control_epsilon_greedy(env, num_episodes=100, epsilon=0.1)


V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")