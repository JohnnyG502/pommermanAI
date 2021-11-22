from keras.models import Model
from keras.layers import Dense, Input
import numpy as np


def initialize_nn():
    
    input_layer = Input(shape=(42,))
    
    X = Dense(units=32,activation='relu')(input_layer)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=16,activation='relu')(X)
    X = Dense(units=12,activation='relu')(X)
     
    # value
    value_output = Dense(units=1, activation='tanh', name='value_output')(X)
     
    # policy
    policy_output = Dense(units=6, activation='softmax', name='policy_output')(X)
     
    model = Model(inputs=input_layer, outputs=[value_output, policy_output])
    
    model.summary()
    
    model.save("nn")

def train(model,embedding_list,action_list,reward_list):
    
    y_policy = np.zeros((embedding_list.shape[0],6))
    
    for row_index in range(y_policy.shape[0]):
        y_policy[row_index][int(action_list[row_index])] = 1 #reward_list[row_index] 
    
    model.fit(embedding_list,[reward_list,y_policy],batch_size = 32, epochs = 100, verbose = 1)
    
#initialize_nn()