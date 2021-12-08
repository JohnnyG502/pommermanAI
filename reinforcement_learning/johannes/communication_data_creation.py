import numpy as np


data = np.load("./states.npy", allow_pickle=True)

true_obs = data.copy()

'''
create last state, current state
'''

counter = 0

agents = [10, 11, 12, 13]
obs_arr = []
for game in true_obs:
    game_arr = []
    for index, step in enumerate(game):
        step_arr = []
        agents_pos = {i: "_" for i in range (10, 14)}
        alive = step[0]["alive"]
        team_1 = False # 10 and 12
        team_2 = False # 11 and 13
        if len(alive) == 4: team_1, team_2 = True 
        elif 10 and 12 in alive: team_1 = True
        elif 11 and 13 in alive: team_2 = True
        if team_1 or team_2:
            for value in alive:
                agents_pos[value] = step[value-10]["position"]
            for value in alive:
                


'''
            if index + 10 in obs["alive"]:
                it = np.nditer(obs["board"], flags=["multi_index"])
                for obs_val in it:
                    if obs_val in agents:
                        agents_pos[obs_val] = it.multi_index
        for obs in step:
            if 


'''

while counter < len(data):
    pos = [-1] * 4
    temp = data[counter:counter+4]
    for i in range(counter, counter+4):
        for index, val in enumerate(data[i]):
            if val in agents:
                temp[:, index] = val      
    optimal_observation[counter:counter+4, :] = temp
    counter += 4
    
np.save("optimal_observations.npy",optimal_observation)




    

    
    

