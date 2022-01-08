'''An example to show how to set up an pommerman game programmatically'''
import numpy as np

import pommerman
from pommerman import agents
import random


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    agent_list = [
        agents.Agent007(),
        agents.SimpleAgent(),
        agents.Agent007(),
        agents.SimpleAgent(),
    ]
    agent_pos = 0

    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    # Run the episodes just like OpenAI Gym
    num_episodes = 1000
    wins = 0
    wins_with_attack = 0
    eps_with_attack = 0
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode + 1))
        if info['result'].value == 0 and agent_pos in info['winners']:
            win = 1
        else:
            win = 0
        for agent in agent_list:
            if type(agent) == pommerman.agents.Agent007:
                if agent.game_with_attack:
                    eps_with_attack += 1
                    if win == 1:
                        wins_with_attack += 1
        wins += win
        print("winrate: " + str(wins/(i_episode + 1)))
        if eps_with_attack > 0:
            print("winrate_attacks: " + str(wins_with_attack / eps_with_attack) + " (" + str(eps_with_attack) + ")")
    env.close()

    print("final winrate: " + str(wins/num_episodes))

    return wins/num_episodes


if __name__ == '__main__':
    main()
