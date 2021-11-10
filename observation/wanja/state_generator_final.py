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
    states = np.empty((number_of_states_to_generate + 50000,38*81))
    counter = 0

    # Run the episodes just like OpenAI Gym
    while counter < number_of_states_to_generate:
        state = env.reset()
        done = False
        while not done:
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            # new part
            state_as_arrays = as_arrays(state)
            states[counter:counter+4] = state_as_arrays
            counter += 4
        print(counter)
    env.close()
    states = states[:number_of_states_to_generate]
    np.save("states.npy",states)


# shifts to position and gives a 9x9 array
def center(board, pos):
    
    result = np.empty((9,9))
    
    board = np.pad(board, ((4, 4), (4, 4)), mode='constant')
    
    result = board[pos[0]:pos[0] + 9,pos[1]:pos[1] + 9]
    
    return result

# splits board with entries between 0 and entries into entries-many boards with entries 0 or 1
def split(board,entries):
    
    new_board = np.zeros(entries*81)
        
    for board_index in range(81):
        type_index = int(board[board_index])
        new_board[81*type_index + board_index] = 1
    
    return new_board

# returns the five 11x11 maps of the four agents centered, splitted and flattened into four 38*81 arrays
# additional information from the states like messages, living agents, ammo etc is NOT considered and has to be added to the embeddings
def as_arrays(state):
    
    state_as_arrays = np.zeros((4,38*81))
    
    for agent in range(4):
        
        pos = state[agent]['position']
        
        board = center(state[agent]['board'],pos).flatten()
            
        state_as_arrays[agent][:14*81] = split(board,14)
        
        bomb_blast_strength = center(state[agent]['bomb_blast_strength'],pos).flatten()
        
        state_as_arrays[agent][14*81:19*81] = split(bomb_blast_strength,5)
        
        bomb_life = center(state[agent]['bomb_life'],pos).flatten()
        
        state_as_arrays[agent][19*81:29*81] = split(bomb_life,10)
        
        bomb_moving_direction = center(state[agent]['bomb_moving_direction'],pos).flatten()
            
        state_as_arrays[agent][29*81:34*81] = split(bomb_moving_direction,5)

        flame_life = center(state[agent]['flame_life'],pos).flatten()
        
        state_as_arrays[agent][34*81:38*81] = split(flame_life,4)
        
    return state_as_arrays

if __name__ == '__main__':
    main()
