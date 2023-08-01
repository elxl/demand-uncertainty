import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_LSTM(nn.Module):
    def __init__(self, n_features, n_time, hid_fc, hid_g, hid_l, dist, device, dropout=0.5):
        super(NN_LSTM,self).__init__()

        self.features = n_features
        self.ntime = n_time
        self.hid_fc = hid_fc
        self.hid_g = hid_g
        self.hid_l = hid_l
        self.dist = dist
        self.dropout = dropout

        self.batchnorm = nn.BatchNorm1d(num_features=n_time)

        # Fully connected layers
        self.fc1 = nn.Linear(n_features, hid_fc)
        self.fc2 = nn.Linear(hid_fc, hid_g)

        # LSTM
        self.lstm = nn.LSTM(hid_g, hid_l)

        # Fully connected layers
        self.fc3 = nn.Linear(hid_l, hid_fc)
        self.fc4 = nn.Linear(hid_fc, hid_fc)

        self.final = nn.Linear(hid_fc, 2)

    def forward(self, x):

        x = self.batchnorm(x)

        # Apply fully connected layers
        x = F.relu(self.fc2(F.dropout(F.relu(self.fc1(x)), self.dropout, training=self.training)))

        # Apply LSTM layer
        out, _ = self.lstm(x)
        out = torch.squeeze(out[:,-1,:]) # only take the last timestep

        # Apply final fully connected layer
        out = F.relu(self.fc4(F.dropout(F.relu(self.fc3(out)), self.dropout, training=self.training)))
        out = self.final(out)

        out_mean = F.softplus(out[:,:,0])
        out_var = F.softplus(out[:,:,1])
        current_out = torch.cat((out_mean, out_var), 0)
        return current_out