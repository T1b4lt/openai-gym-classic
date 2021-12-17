import argparse
import yaml
import gym
from agents.q_agent import QAgent
from utils import utils
from utils.discretizator import Discretizator


def main(args):

    f = open(args.configfile, "r")
    config = yaml.load(f, Loader=yaml.FullLoader)

    N_EPISODES = config['n_episodes']
    N_STEPS = config['n_steps']
    EXPLORATION_RATIO = config['exploration_ratio']
    LEARNING_RATE = config['learning_rate']
    DISCOUNT_FACTOR = config['discount_factor']
    E_DECAY_LIMIT = config['e_decay_limit']
    E_DECAY_RATE = config['e_decay_rate']
    BINS_POS = config['bins_pos']
    BINS_VEL = config['bins_vel']
    RENDER = config['render']
    REPORT_FILE = config['report_file']

    print("\n################ Parameters ################\n")
    print("N_EPISODES:", N_EPISODES)
    print("N_STEPS:", N_STEPS)
    print("EXPLORATION_RATIO:", EXPLORATION_RATIO)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("DISCOUNT_FACTOR:", DISCOUNT_FACTOR)
    print("RENDER:", RENDER)
    print("\n############################################\n")

    env = gym.make('MountainCar-v0')

    # TODO: Tengo que ver cuales son los nombres de las acciones
    actions_dict = {0: 'Zero', 1: 'One', 2: 'Two'}
    hist = {}

    discretizator = Discretizator(
        env.observation_space.low, env.observation_space.high, bins_array=[BINS_POS, BINS_VEL])

    agent = QAgent(discretizator.get_n_states(), env.action_space, exploration_ratio=EXPLORATION_RATIO,
                   learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR, e_decay_limit=E_DECAY_LIMIT, e_decay_rate=E_DECAY_RATE)

    print("\n\n############### Ini Training ###############\n")
    for i_episode in range(N_EPISODES):
        state = env.reset()
        reward_counter = 0
        if RENDER:
            print("############### Ini Episode", i_episode, "###############")
        for t in range(N_STEPS):
            if RENDER:
                env.render()
                print("Actual State:", state)
            action = agent.get_next_step(discretizator.idx_state(state))
            if RENDER:
                print("Action:", actions_dict[action])
            next_state, reward, done, info = env.step(action)
            reward_counter += reward
            if RENDER:
                print("Next State:", next_state, "\n")
            agent.update_qtable(discretizator.idx_state(
                state), action, reward, discretizator.idx_state(next_state), done)
            state = next_state
            if done:
                break
        if i_episode % 10 == 0:
            print('Episode: {} Reward: {} Steps Taken: {} Info: {}'.format(
                i_episode, reward_counter, t+1, info))
        hist[i_episode] = {'reward': reward_counter, 'steps': t+1}
        if RENDER:
            print("############### End Episode", i_episode, "###############")
    print("\n############### End Training ###############\n")
    print("\n\n################## Report ##################\n")
    report = {"average_reward": utils.get_average_reward_last_n(hist, N_EPISODES),
              "average_reward_last_10": utils.get_average_reward_last_n(hist, int(N_EPISODES*0.1)),
              "average_steps": utils.get_average_steps_last_n(hist, N_EPISODES),
              "average_steps_last_10": utils.get_average_steps_last_n(hist, int(N_EPISODES*0.1))
              }
    print("Average reward:", report["average_reward"])
    print("Average reward of last 10%("+str(int(N_EPISODES*0.1))+"):",
          report["average_reward_last_10"])
    print("Average steps:", report["average_steps"])
    print("Average steps of last 10%("+str(int(N_EPISODES*0.1))+"):",
          report["average_steps_last_10"])
    print("\nQ-table:")
    print(agent.qtable)
    print("\n################ End Report ################")
    if REPORT_FILE:
        utils.generate_report_file(config, report, hist, agent.qtable)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test argparse')
    parser.add_argument('-f', '--file', help='agent config file',
                        required=True, type=str, dest='configfile')
    args = parser.parse_args()
    main(args)
