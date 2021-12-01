
import pandas as pd
import numpy as np

class CommunicationProtocol():
    def __init__(self, infos):
        #self.enemy_1_pos = enemy_1
        #self.enemy_2_pos = enemy_2

        #self.me_pos = me

        self.message2value = {
            1: {1: [0,0,0],2: [0,1,0],3: [0,2,0],4: [0,3,0],5: [0,4,0],6: [0,0,1],7: [1,0,0],8: [1,1,0]}, 
            2: {1: [1,2,0],2: [1,3,0],3: [1,4,0],4: [1,1,1],5: [2,0,0],6: [2,1,0],7: [2,2,0],8: [2,3,0]}, 
            3: {1: [2,4,0],2: [2,2,1],3: [3,0,0],4: [3,1,0],5: [3,2,0],6: [3,3,0],7: [3,4,0],8: [3,3,1]}, 
            4: {1: [4,0,0],2: [4,1,0],3: [4,2,0],4: [4,3,0],5: [4,4,0],6: [4,4,1],7: [0,1,1],8: [0,2,1]}, 
            5: {1: [0,3,1],2: [0,4,1],3: [1,2,1],4: [1,3,1],5: [1,4,1],6: [2,0,1],7: [2,1,1],8: [2,3,1]}, 
            6: {1: [2,4,1],2: [3,0,1],3: [3,1,1],4: [1,0,1],5: [3,2,1],6: [3,4,1],7: [4,0,1],8: [4,1,1]}, 
            7: {1: [4,2,1],2: [4,3,1]}}

        self.value2message = {}
        for key in self.message2value.keys():
            for key2 in self.message2value[key].keys():
                self.value2message[tuple(self.message2value[key][key2])] = (key, key2)

    def messageToPosition(self, message):
        return tuple(self.message2value[message[0]][message[1]])

    def PositionToMessage(self, position):
        return self.value2message[tuple(position)]

    def positions(self, e1, e2, me):
        com_vals = [0, 0, 0]
        for index, obj in enumerate(list((e1, e2, me))):
            # quadrant 1: (0-5, 0-4)
            # quadrant 2: (0-5, 5-10)
            # quadrant 3: (6-10, 5-10)
            # quadrant 4: (6-10, 0-4)

            # not visible: 0

            # me_top: 0
            # me_bot: 1

            if obj[0] < 5:
                if obj[1] > 5: com_vals[index] = 3
                else : com_vals[index] = 4
                if index == 2: com_vals[index] = 0
            else:
                if obj[1] > 5: com_vals[index] = 2
                else: 
                    com_vals[index] = 1
                if index == 2: com_vals[index] = 1
        return com_vals

    def interpretPositions(self, state, friend_pos):
        # standart gamestate
        # [x,y,z] friend info
        pass

    def smartRules(self, pos_arr_me, pos_arr_team):
        if pos_arr_me == pos_arr_team:  # no state when array same
            pass
        elif pass_arr_me[:2] == 0 and pos_arr_team[:2] == 0: # change state when enemy arrays same but pos of team changes
            # if new friend pos != old friend pos: 
            pass





x = CommunicationProtocol("lol")
print(x.value2message)
