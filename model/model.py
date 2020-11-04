import torch
import torch.nn as nn
import torch.nn.functional as F

class VFWS(nn.Module):
    def __init__(self):
        super(VFWS,self).__init__()
        self.conv = nn.Sequential(
                # cnn1
                nn.ZeroPad2d((3, 3, 0, 0)),
                nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn2
                nn.ZeroPad2d((0, 0, 3, 3)),
                nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn3
                nn.ZeroPad2d(2),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn4
                nn.ZeroPad2d((2, 2, 4, 4)),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)), # (9, 5)
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn5
                nn.ZeroPad2d((2, 2, 8, 8)),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)), # (17, 5)
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn6
                nn.ZeroPad2d((2, 2, 16, 16)),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)), # (33, 5)
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn7
                nn.ZeroPad2d((2, 2, 32, 32)),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)), # (65, 5)
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn8
                nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)), 
                nn.BatchNorm2d(8), nn.ReLU(),
                )
        #TODO fix args
        self.lstm = nn.LSTM(
                8*hp.audio.num_freq,
                hp.model.lstm_dim,
                batch_first=True,
                bidirectional=True)

        self.fc1 = nn.Linear(2*hp.model.lstm_dim, hp.model.fc1_dim)
        self.fc2 = nn.Linear(hp.model.fc1_dim, hp.model.fc2_dim)      

