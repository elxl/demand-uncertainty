import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNLayer
from torch_geometric.nn import GCNConv

class GCN_LSTM(nn.Module):
    def __init__(self, n_features, n_stations, hid_g, hid_fc, hid_l, meanonly, homo, nadj, dist, device, dropout=0.5):
        super(GCN_LSTM, self).__init__()

        self.n_features = n_features
        self.n_stations = n_stations
        self.n_hid_g = hid_g
        self.n_hid_fc = hid_fc
        self.n_hid_l = hid_l
        self.dropout = dropout
        self.meanonly = meanonly
        self.homo = homo
        self.nadj = nadj
        self.dist = dist

        ###################################
        #-------- Encoding layers --------#
        ###################################
        self.batchnorm = nn.BatchNorm1d(num_features=n_stations)

        # GCN layer
        self.gcn = GCNLayer(n_features, hid_g, hid_g, dropout)

        # Linear layers
        self.fc1 = nn.Linear(n_stations * hid_g, hid_fc)
        self.fc2 = nn.Linear(hid_fc, hid_g)

        # LSTM layer
        self.lstm = nn.LSTM(hid_g, hid_l)

        # Linear layers
        self.fc3 = nn.Linear(hid_l, hid_fc)
        self.fc4 = nn.Linear(hid_fc, hid_fc)

        ####################################
        #-------- Component layers --------#
        ####################################

        # layers bringing everything together
        if meanonly | (homo>0) | dist in ['poission']:
            mult = 1
        elif dist in ['norm','nb','zipoission','tnorm','lognorm']:
            mult = 2
        elif dist in ['zinb']:
            mult = 3
        self.mult = mult
        self.final = nn.Linear(hid_fc, n_stations*mult)

        # History
        self.recent_on_history_mean = nn.Linear(hid_fc, n_stations)

        # # Weather
        # self.weather_weights_mean = nn.Parameter(torch.rand((n_time, 2*n_stations)))

        # # Level of Service
        # self.los_weights_mean = nn.Parameter(torch.rand(n_time, n_stations))

        if dist in ['norm','nb','zipoission','tnorm','lognorm']:
            self.recent_on_history_var = nn.Linear(hid_fc, n_stations)
            # self.weather_weights_var = nn.Parameter(torch.rand((n_time, 2*n_stations)))
        if dist in ['zinb']:
            self.recent_on_history_pi = nn.Linear(hid_fc, n_stations)

    def forward(self, x, adj, history, weather, los, device):
        """
        qod: batch_size,n_time
        xs: batch_size,ndemo
        """

        batch_size, timesteps, stations, features = x.size()

        ############################
        #-------- Encoding --------#
        ############################

        # Convert input shape
        x = x.view(batch_size*timesteps, stations, features)
        x = self.batchnorm(x)

        # Apply graph convolution layer
        out = self.gcn(x, adj, device)

        # Concatenate stations for timesteps
        out = out.view(batch_size, timesteps, -1) 

        # Apply fully connected layers
        l_in = F.relu(self.fc2(F.dropout(F.relu(self.fc1(out)), self.dropout, training=self.training)))

        # Apply LSTM layer
        out, _ = self.lstm(l_in)
        out = torch.squeeze(out[:,-1,:]) # only take the last timestep

        # Apply final fully connected layer
        out = l_in = F.relu(self.fc4(F.dropout(F.relu(self.fc3(out)), self.dropout, training=self.training)))

        ####################################
        #-------- Component layers --------#
        ####################################

        # Layer of everything
        gl_out = self.final(out)

        # History and weather
        recent_on_history_weights_mean = torch.sigmoid(self.recent_on_history_mean(out)).view(batch_size, stations)
        history = torch.squeeze(history)
        history_mean = history * recent_on_history_weights_mean  

        # weather = weather.view(batch_size, stations, 2)
        # weather_mean = self.weather_weights_mean 

        if self.dist in ['norm','nb','zipoission','tnorm','lognorm']:
            recent_on_history_weights_var = torch.sigmoid(self.recent_on_history_var(out)).view(batch_size, stations)
            history_var = history * recent_on_history_weights_var
            # weather_var = torch.squeeze(torch.bmm(weather, torch.mm(qod, self.weather_weights_var).view(batch_size, 2, stations)))
        if self.dist in ['zinb']:
            recent_on_history_weights_pi = torch.sigmoid(self.recent_on_history_pi(out)).view(batch_size, stations)
            history_pi = history * recent_on_history_weights_pi           

        # Level of Service
        # los = torch.squeeze(los)
        # los_mean = los * torch.mm(qod, self.los_weights_mean)

        # Combine everything
        if (self.meanonly)|(self.homo>0)|(self.dist in ['poission']):
            gl_out = gl_out.view(batch_size, -1, 1)
            out_mean = F.softplus(gl_out[:,:,0]+history_mean)
            return out_mean
        elif self.dist in ['norm','nb','zipoission','tnorm','lognorm']:
            gl_out = gl_out.view(batch_size, -1, 2)
            out_mean = F.softplus(gl_out[:,:,0]+history_mean)
            if self.dist in ['norm','tnorm','lognorm']:
              out_var = F.softplus(gl_out[:,:,1]+history_var)
            elif self.dist in ['nb','zipoission']:
              out_var = F.sigmoid(gl_out[:,:,1]+history_var) - 1e-5

            current_out = torch.cat((out_mean, out_var), 0)
            return current_out
        else:
            gl_out = gl_out.view(batch_size, -1, 3)
            out_mean = F.softplus(gl_out[:,:,0]+history_mean)
            out_var = F.softplus(gl_out[:,:,1]+history_var)
            out_pi = F.sigmoid(gl_out[:,:,2]+history_pi)
            current_out = torch.cat((out_mean, out_var, out_pi), 0)
            return current_out          
