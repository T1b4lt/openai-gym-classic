import gym
from agents.q_agent import QAgent
from utils import utils

N_EPISODES = 10
N_STEPS = 100
RENDER = True

env = gym.make('MountainCar-v0')

actions_dict = {0: 'Zero', 1: 'One', 2: 'Two'}
hist = {}


for i_episode in range(N_EPISODES):
    state = env.reset()
    if RENDER:
        print("############### Ini Episode", i_episode, "###############")
    for t in range(N_STEPS):
        if RENDER:
            env.render()
            print("Actual State:", state)
        action = env.action_space.sample()
        if RENDER:
            print("Action:", actions_dict[action])
        next_state, reward, done, info = env.step(action)
        if RENDER:
            print("Next State:", next_state, "\n")
        state = next_state
        if done:
            break
    if i_episode % 10 == 0:
        print('Episode: {} Reward: {} Steps Taken: {} Info: {}'.format(
            i_episode, reward, t+1, info))
    hist[i_episode] = {'reward': reward,
                       'steps': t+1}
    if RENDER:
        print("############### End Episode", i_episode, "###############")
print("Average reward:", utils.get_average_reward_last_n(hist, N_EPISODES))
print("Average reward of last 100:", utils.get_average_reward_last_n(hist, 100))
print("Average steps:", utils.get_average_steps_last_n(hist, N_EPISODES))
print("Average steps of last 100:", utils.get_average_steps_last_n(hist, 100))
env.close()
