'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
import keras
import nn


def main():
    '''
        Schritt 1: Generiere states mit "state_generator_final"
        Schritt 2: Trainiere den encoder mit "autoencoder_final"
        Alternativ zu 1 + 2: Nutze einen vortrainierten encoder
        Schritt 3: Nutze dieses script zum Training eines RL-Netzes
    '''

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    
    # load encoder
    encoder = keras.models.load_model("encoder")
    
    # load and compile model
    model = keras.models.load_model("nn")
    
    model.compile(optimizer='adam',
              loss={'value_output': 'mse', 'policy_output': 'categorical_crossentropy'},
              loss_weights={'value_output': 1., 'policy_output': 1.})
              #metrics={'value_output': 'mse', 'policy_output': 'cosine_similarity'})
    
    # number of game episodes
    episodes = 1
    
    embedding_list = np.zeros((0,42))
    action_list = np.zeros((0,1))
    reward_list = np.zeros((0,1))
    
    # Run the episodes
    for i_episode in range(episodes):
        new_embeddings,new_actions,new_rewards = gamerun(env,encoder)
        embedding_list = np.r_[embedding_list,new_embeddings]
        action_list = np.r_[action_list,new_actions]
        reward_list = np.r_[reward_list,new_rewards]
        print('Episode {} finished'.format(i_episode))
        
    nn.train(model,embedding_list,action_list,reward_list)
    env.close()
    
    model.save("nn")

def gamerun(env,encoder):
    
    embedding_list = np.zeros((900*4,42))
    action_list = np.zeros((900*4,1))
    reward_list = np.zeros((900*4,1))
    
    state = env.reset()
    
    done = False
    
    counter = 0
    while not done:
        # do not render the game
        # env.render()
        actions = env.act(state)
        embedding_list[counter:counter+4,:] = state_to_embedding(state,encoder)
        action_list[counter:counter+4,:] = np.array(actions).reshape(4,1)
        state, reward, done, info = env.step(actions)
        reward_list[counter:counter+4,:] = np.array(reward).reshape(4,1)
        counter += 4
        
    return embedding_list[:counter],action_list[:counter],reward_list[:counter]

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

def additional_information(state):
    
    additional_information = np.zeros((4,10))
    
    for agent in range(4):
        if 10 in state[agent]['alive']:
            additional_information[0] = 1
        if 11 in state[agent]['alive']:
            additional_information[1] = 1
        if 12 in state[agent]['alive']:
            additional_information[2] = 1
        if 13 in state[agent]['alive']:
            additional_information[3] = 1
            
        additional_information[agent][4] = state[agent]['ammo']
        additional_information[agent][5] = state[agent]['blast_strength']
        additional_information[agent][6] = int(state[agent]['can_kick'])
        additional_information[agent][7] = state[agent]['position'][0]
        additional_information[agent][8] = state[agent]['position'][1]
        additional_information[agent][9] = state[agent]['step_count']/400 # scaling for same order
        
    return additional_information

def state_to_embedding(state, encoder):
    
    boards_as_arrays = as_arrays(state)
    
    embedded_boards = encoder.predict(boards_as_arrays)
    
    ai = additional_information(state)
    
    embedding = np.c_[embedded_boards,ai]
    
    return embedding

if __name__ == '__main__':
    main()
