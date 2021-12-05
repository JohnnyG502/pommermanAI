import numpy as np
import pandas as pd

data = np.load("./states.npy")

'''
create last state, current state
'''

counter = 0

optimal_observation = np.zeros((len(data), len(data[0])))


agents = [10, 11, 12, 13]

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




    

    
    

