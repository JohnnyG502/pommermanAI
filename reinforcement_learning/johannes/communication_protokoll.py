
import pandas as pd
import numpy as np

class CommunicationProtocol():
    def __init__(self):
        self.message2value = {
            1: {1: [0,0,5],2: [0,1,5],3: [0,2,5],4: [0,3,5],5: [0,4,5],6: [0,0,6],7: [1,0,5],8: [1,1,5]}, 
            2: {1: [1,2,5],2: [1,3,5],3: [1,4,5],4: [1,1,6],5: [2,0,5],6: [2,1,5],7: [2,2,5],8: [2,3,5]}, 
            3: {1: [2,4,5],2: [2,2,6],3: [3,0,5],4: [3,1,5],5: [3,2,5],6: [3,3,5],7: [3,4,5],8: [3,3,6]}, 
            4: {1: [4,0,5],2: [4,1,5],3: [4,2,5],4: [4,3,5],5: [4,4,5],6: [4,4,6],7: [0,1,6],8: [0,2,6]}, 
            5: {1: [0,3,6],2: [0,4,6],3: [1,2,6],4: [1,3,6],5: [1,4,6],6: [2,0,6],7: [2,1,6],8: [2,3,6]}, 
            6: {1: [2,4,6],2: [3,0,6],3: [3,1,6],4: [1,0,6],5: [3,2,6],6: [3,4,6],7: [4,0,6],8: [4,1,6]}, 
            7: {1: [4,2,6],2: [4,3,6]}}

        self.value2message = {}
        for key in self.message2value.keys():
            for key2 in self.message2value[key].keys():
                self.value2message[tuple(self.message2value[key][key2])] = (key, key2)

    def messageToPosition(self, message):
        return self.message2value[message[0]][message[1]]

    def PositionToMessage(self, position):
        #print(position)
        return list(self.value2message[tuple(position)])



class PositionDefinition():
    def __init__(self):
        self.field_size = 11

    def QuadrantToObeservationArr(self, quadrant):
        enemy_1 = np.zeros((self.field_size, self.field_size))
        enemy_2 = np.zeros((self.field_size, self.field_size))
        me = np.zeros((self.field_size, self.field_size))

        observation_list = [enemy_1, enemy_2, me]
        
        for index, quad in enumerate(quadrant):
            # quadrant (0-5, 0-4): 1
            # quadrant (0-5, 5-10): 2
            # quadrant (6-10, 5-10): 3
            # quadrant (6-10, 0-4): 4

            # not visible: 0

            # me_top: 0
            # me_bot: 1

            if quad == 1:
                observation_list[index][:6, :5] = 1
            elif quad == 2:
                observation_list[index][:6, 5:] = 1
            elif quad == 3:
                observation_list[index][6:, 5:] = 1
            elif quad == 4:
                observation_list[index][6:, :5] = 1
            elif quad == 5:
                observation_list[index][:6, :] = 1
            elif quad == 6:
                observation_list[index][6:, :] = 1
        
        return observation_list


    def PosTupleToQuadrant(self, e1, e2, me):
        #print(me)
        com_vals = [0, 0, 0]
        for index, obj in enumerate(list((e1, e2, me))):
            # quadrant (0-5, 0-4): 1
            # quadrant (0-5, 5-10): 2
            # quadrant (6-10, 5-10): 3
            # quadrant (6-10, 0-4): 4

            # not visible: 0

            # me_top: 5
            # me_bot: 6

            if type(obj) is tuple: 
                if obj[0] < 5:
                    if obj[1] > 5: com_vals[index] = 3
                    elif obj[1] <= 5: com_vals[index] = 4
                    if index == 2: com_vals[index] = 5
                elif obj[0] >= 5:
                    if obj[1] > 5: com_vals[index] = 2
                    elif obj[1] <= 5: com_vals[index] = 1
                    if index == 2: com_vals[index] = 6
        return com_vals

    def ObservationArrToPosTuple(self, observation, enemies):
        e1_pos = -1
        e2_pos = -1
        it = np.nditer(observation, flags=["multi_index"])
        for obs_val in it:
            if obs_val in enemies:
                if obs_val == max(enemies): e2_pos = it.multi_index
                elif obs_val == min(enemies): e1_pos = it.multi_index
        return e1_pos, e2_pos







