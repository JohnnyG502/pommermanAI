'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    
    # Make a list of states as numeric arrays to train encoder-decoder
    number_of_states_to_generate = 50000
    states = []
    counter = 0

    # Run the episodes just like OpenAI Gym
    while counter < number_of_states_to_generate:
        state = env.reset()
        done = False
        game = []
        while not done:
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            # new part
            game.append(state)
            counter += 4
        print(counter)
        states.append(game)
    env.close()
    np.save("states.npy", np.array(states, dtype="object"), allow_pickle=True)



# returns the five 11x11 maps of the four agents centered, splitted and flattened into four 38*81 arrays
# additional information from the states like messages, living agents, ammo etc is NOT considered and has to be added to the embeddings
def as_arrays(state):
    
    state_as_arrays = np.zeros((4,11*11))
    
    for agent in range(4):
        
        pos = state[agent]['position']
        
        board = state[agent]['board'].flatten()

        state_as_arrays[agent] = board
            
        
        
    return state_as_arrays



if __name__ == '__main__':
    main()
