import numpy as np
import pandas as pd
#from collections import defaultdict

from communication_protokoll import CommunicationProtocol, PositionDefinition

data = np.load("./states.npy", allow_pickle=True)

true_obs = data.copy()

'''
create last state, current state
'''

counter = 0

agents = [10, 11, 12, 13]
obs_arr = []

dataset = pd.DataFrame(columns=["current_obs", "last_obs", "current_true_obs", "last_true_obs", "message"])
transformator = PositionDefinition()
translator = CommunicationProtocol()


for game in true_obs:
    game_arr = []
    last_obs = {}
    last_perf_obs = {}
    for index, step in enumerate(game):
        step_arr = []
        agents_pos = {i: "_" for i in range (10, 14)}
        alive = step[0]["alive"]
        team_1 = False # 10 and 12
        team_2 = False # 11 and 13
        if len(alive) == 4: team_1, team_2 = True, True 
        elif 10 and 12 in alive: team_1 = True
        elif 11 and 13 in alive: team_2 = True
        if team_1 or team_2:
            for value in alive:
                agents_pos[value] = step[value-10]["position"]

            if team_1:
                current_1 = step[0]["board"]
                e1_pos, e2_pos = transformator.ObservationArrToPosTuple(current_1, [11, 13])
                msg_1 = translator.PositionToMessage(transformator.PosTupleToQuadrant(e1_pos, e2_pos, agents_pos[10]))
                current_2 = step[2]["board"]
                e1_pos, e2_pos = transformator.ObservationArrToPosTuple(current_2, [11, 13])
                msg_2 = translator.PositionToMessage(transformator.PosTupleToQuadrant(e1_pos, e2_pos, agents_pos[12]))

                current_1_true, current_2_true = current_1.copy(), current_2.copy()
                for key in agents_pos.keys():
                    current_1_true[agents_pos[key][0], agents_pos[key][1]] = key
                    current_2_true[agents_pos[key][0], agents_pos[key][1]] = key
                

                if index != 0:
                    # integrate last perfect obs?
                    past_1 = last_obs[10]
                    past_2 = last_obs[12]
                    past_1_true = last_perf_obs[10]
                    past_2_true = last_perf_obs[12]
                elif index == 0:
                    past_1 = current_1
                    past_2 = current_2 
                    past_1_true = current_1_true
                    past_2_true = current_2_true

                last_obs[10] = current_1
                last_obs[12] = current_2
                last_perf_obs[10] = current_1_true
                last_perf_obs[12] = current_2_true

                temp_sample_1 = {"current_obs": current_1, "last_obs": past_1, "current_true_obs": current_1_true, "last_true_obs": past_1_true, "message": msg_1}
                temp_sample_2 = {"current_obs": current_2, "last_obs": past_2, "current_true_obs": current_2_true, "last_true_obs": past_2_true, "message": msg_2}
                dataset = dataset.append(temp_sample_1, ignore_index=True)
                dataset = dataset.append(temp_sample_2, ignore_index=True)
                
            if team_2:

                current_3 = step[1]["board"]
                e3_pos, e4_pos = transformator.ObservationArrToPosTuple(current_3, [10, 12])
                msg_3 = translator.PositionToMessage(transformator.PosTupleToQuadrant(e3_pos, e4_pos, agents_pos[11]))
                
                current_4 = step[3]["board"]
                e3_pos, e4_pos = transformator.ObservationArrToPosTuple(current_4, [10, 12])
                msg_4 = translator.PositionToMessage(transformator.PosTupleToQuadrant(e3_pos, e4_pos, agents_pos[13]))

                current_3_true, current_4_true = current_3.copy(), current_4.copy()
                for key in agents_pos.keys():
                    current_3_true[agents_pos[key][0], agents_pos[key][1]] = key
                    current_4_true[agents_pos[key][0], agents_pos[key][1]] = key

                if index != 0:
                    # integrate last perfect obs?
                    past_3 = last_obs[11]
                    past_4 = last_obs[13]
                    past_3_true = last_perf_obs[11]
                    past_4_true = last_perf_obs[13]
                elif index == 0:
                    past_3 = current_3
                    past_4 = current_4 
                    past_3_true = current_3_true
                    past_4_true = current_4_true

                last_obs[11] = current_3
                last_obs[13] = current_4
                last_perf_obs[11] = current_3_true
                last_perf_obs[13] = current_4_true

                temp_sample_1 = {"current_obs": current_3, "last_obs": past_3, "current_true_obs": current_3_true, "last_true_obs": past_3_true, "message": msg_3}
                temp_sample_2 = {"current_obs": current_4, "last_obs": past_4, "current_true_obs": current_4_true, "last_true_obs": past_4_true, "message": msg_4}
                dataset = dataset.append(temp_sample_1, ignore_index=True)
                dataset = dataset.append(temp_sample_2, ignore_index=True)

dataset.to_csv("msg_pred_data.csv")

    

    
    

