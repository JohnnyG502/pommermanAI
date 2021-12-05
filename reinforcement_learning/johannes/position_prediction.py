import numpy as np
import pandas as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

class PositionPrediction(nn.Module):
    def __init__(self, input_dim, current_obs, decoded_message):
        super(PositionPrediction. self).__init__()

        self.input_dim = input_dim



    def forward(self, previous_obs, current_obs, decoded_message):
        return