import numpy as np
import pandas as pd
import time, random, math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

# modules
from communication_protokoll import CommunicationProtocol, PositionDefinition
from utils import *

import ast 


class PositionPrediction(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim
        
        self.cnn1 = nn.Conv2d(input_dim, 7, 2, stride=1)
        self.cnn2 = nn.Conv2d(5, 1, 2, stride=1)
        #self.l1 = nn.Linear()
        
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, obs):
        
        out = self.relu(self.cnn1(obs))
        out = self.relu(self.cnn2(out))
        #out = self.sig(self.l1(out))
        
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
            print(batch, input)
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

class createTrainingData():
    pass

class Preprocessing():
    def __init__():
        pass

    

class Dataset(Dataset):
    def __init__(self, data, labels, elements):
        self.data = data
        self.label = labels
        self.elements = elements


    def transformation(self, last_obs, current_obs, message):
        """
        transforms from 3 types of observations (last, current, message) into an (X,Y,Z) array of binary encodings
        """
        message_obs = transformator.QuadrantToObeservationArr(translator.messageToPosition(message))
        last_obs, current_obs, message_obs = binaryEncoding(last_obs, self.elements), binaryEncoding(current_obs, self.elements), binaryEncoding(message_obs, self.elements)
        return np.concatenate((last_obs, current_obs, message_obs), axis=2)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        data = data.apply(lambda row: self.transformation(row["current_obs"], row["last_obs"], row["message"]), axis=1)
        return torch.tensor(data), torch.tensor(label)

def dumbConvFunction(col):
    new_col = []
    for array in col:
        s_arr = array.split("\r\n")
        s_arr[0] = s_arr[1:]
        
    return
    
    
def main():
    
    param = {
            "epochs": 100,
            "batch_size": 10,
            "l_rate": 0.001,
            "path": "./saved_models/model_weights.pt",
            "test_size": 0.5
            }

    transformator = PositionDefinition()
    translator = CommunicationProtocol()

    elements = [1, 2, 5, 10, 11, 12, 13]


    data = pd.read_csv(".\..\..\msg_pred_data.csv")
    data = data.drop(columns=["Unnamed: 0"], axis=1)
    data = dumbConvFunction(data["current_obs"])
    print(data)
    current_obs = []
    last_obs = []
    data["last_obs"]
    message = []
    data["message"]

    print(message)



    input = data[['current_obs', 'last_obs', 'message']].copy()
    labels = data[["current_true_obs"]]


    X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=param["test_size"])

    train_loader, test_loader = DataLoader(Dataset(X_train, y_train, elements), batch_size=param["batch_size"], shuffle=True), DataLoader(Dataset(X_test, y_test, elements), batch_size=param["batch_size"], shuffle=True)
    

    
    
    model = PositionPrediction(len(elements)*3)
    optimizer = optim.Adam(model.parameters(), lr=param["l_rate"])
    criterion = nn.MSELoss()
    
    training = Training(param["epochs"], param["batch_size"], optimizer, criterion, model)
    
    training.train_setup(data, param["path"])
    training.evaluate(data)

if __name__ == main():
    main()
        
        