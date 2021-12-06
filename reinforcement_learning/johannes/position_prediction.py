import numpy as np
import pandas as pd
import time, random, math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

from sklearn.model_selection import train_test_split

# modules
from communication_protokoll import ComunicationProtocol, PositionDefinition


class PositionPrediction(nn.Module):
    def __init__(self, input_dim, current_obs, last_obs, message_obs):
        super(PositionPrediction. self).__init__()

        self.input_dim = input_dim
        
        self.cnn1 = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        self.cnn2 = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        self.l1 = nn.Linear()
        
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
        


    def forward(self, previous_obs, current_obs, message_obs):
        input = np.append(previous_obs, current_obs, axis=0)
        input = np.append(input, message_obs, axis=0)
        
        out = self.relu(self.cnn1(input))
        out = self.relu(self.cnn2(out))
        out = self.sig(self.l1(out))
        
        return out

class Training():
    def __init__(self, epochs, batch_size, optimizer, criterion, model):
        self.optimizer, self.criterion, self.model = optimizer, criterion, model
        self.epochs, self.batch_size = epochs, batch_size
        
        self.best_train_loss = 0
        
    def evaluate(self, iterator):
        self.model.eval()
        
        epoch_loss = 0
        
        with torch.no_grad():
            for batch, input in enumerate(iterator):
                self.optimizer.zero_grad()
                output = self.model(input)
                
                loss = self.criterion(output, target)
                epoch_loss += loss.item()
            
        test_loss = epoch_loss / len(iterator)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
        
    def train(self, iterator):
        self.model.train()
        
        epoch_loss = 0
        
        for batch, input in enumerate(iterator):
            self.optimizer.zero_grad()
            output = self.model(input)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)
    
    def train_setup(self, iterator, path):
        for epoch in range(self.epochs):
            
            start_time = time.time()
            
            train_loss = self.train(iterator)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                torch.save(self.model.state_dict(), model_path)

            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
def main():
    
    param = {
            "epochs": 100,
            "batch_size": 10,
            "l_rate": 0.001,
            "path": "./saved_models/model_weights.pt"
            }
    
    data = np.load("./states.npy")
    
    # current, previous and optimal observations
    current_obs = data.copy()
    target = np.load("./optimal_observations.npy")
    previous_obs = data.insert(0, data.pop())
    
    # create message based arrays

    
    
    model = PositionPrediction()
    optimizer = nn.Adam(model.parameter(), lr=param["l_rate"])
    criterion = nn.MSELoss()
    
    training = Training(param["epochs"], param["batch_size"], optimizer, criterion, model)
    
    training.train_setup(data)
    training.evaluate(data)



if __name__ == main():
    main()
        
        