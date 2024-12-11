import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils.prepare_data import prepare_input
from utils.class_loss import MVELoss
from model.class_GCN import GCN_LSTM

# Define hyperparameters
n_features = 2
n_stations = 63
hid_g = 64
hid_fc = 64
hid_l = 64
meanonly = False
homo = 0
batch_size = 32
learning_rate = 0.001
weight_decay = 0.001
dropout = 0.2
num_epochs = 10
nadj = ['euc','con','func']
dist = 'tnorm'
z = 0.95

# Define other parameters
SAVEPATH = "weights/model.pt"
filepath = 'data/processed/1906_diff.npz'
adjfile = 'data/processed/adjlist.csv'


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Create the network
net = GCN_LSTM(n_features,n_stations,hid_g,hid_fc,hid_l,meanonly,homo,nadj,dist,device,dropout)
net = net.to(device)

# Define the loss function
loss_fn = MVELoss(dist)

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Read in and prepare dataset
data = np.load(filepath)
x = data['x']
y = data['y']
adj = pd.read_csv(adjfile)
history = data['history']
weather = data['weather']
los = data['los']

train_loader,val_loader,test_loader,adj_torch = prepare_input(x,y,adj,nadj,history,weather,los,device,batch_size=batch_size)
train_num = len(train_loader.dataset)
eval_num = len(val_loader.dataset)
batch_number = len(train_loader)


print('Start training ...')
print(f"Training sample batches:{batch_number}")

# Training loop
loss_history = []
loss_eval = []
for epoch in range(1,num_epochs+1):

    running_loss = 0.0

    # Iterate over the batches
    for i,traindata in enumerate(train_loader,1):

        net.train()

        batch_x, batch_y, batch_history, batch_weather, batch_los = traindata

        batch_x = batch_x.float()
        batch_y = torch.squeeze(batch_y).float()
        batch_history = batch_history.float()

        batch_x, batch_y, batch_history, batch_weather, batch_los = \
                batch_x.to(device), batch_y.to(device), batch_history.to(device), \
                batch_weather.to(device), batch_los.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = net(batch_x, adj_torch, batch_history, batch_weather, batch_los, device)

        # Compute the loss
        if outputs.shape[0]!=net.mult*batch_size:
            if net.mult == 1:
                output_loc = outputs[:,:]
                output_scale = None
                output_pi = None
            elif net.mult == 2:
                batch_new = int((outputs.shape[0])/2)
                output_loc = outputs[:batch_new,:]
                output_scale = outputs[batch_new:,:]
                output_pi = None
            else:
                batch_new = int((outputs.shape[0])/3)
                output_loc = outputs[:batch_new,:]
                output_scale = outputs[batch_new:2*batch_new,:]
                output_pi = outputs[2*batch_new:,:]           
        else:
            if net.mult == 1:
                output_loc = outputs[:,:]
                output_scale = None
                output_pi = None
            elif net.mult == 2:
                output_loc = outputs[:batch_size,:]
                output_scale = outputs[batch_size:,:]
                output_pi = None
            else:
                output_loc = outputs[:batch_size,:]
                output_scale = outputs[batch_size:2*batch_size,:]
                output_pi = outputs[2*batch_size:,:]
        loss = loss_fn(output_loc, output_scale, batch_y,output_pi)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # if i%20 == 0:
        #   print('Ep:', epoch, 'Batch:', i, 'Loss:', loss)
        running_loss += loss.item()

    # Print the average loss for the epoch and evaluate
    loss_history.append(running_loss/train_num)

    ################ Evaluate ###############
    net.eval()

    eval_loss = 0

    for j,evaldata in enumerate(val_loader):
        batch_x, batch_y, batch_history, batch_weather, batch_los = evaldata

        batch_x = batch_x.float()
        batch_y = torch.squeeze(batch_y).float()
        batch_history = batch_history.float()

        batch_x, batch_y, batch_history, batch_weather, batch_los = \
            batch_x.to(device), batch_y.to(device), batch_history.to(device), \
            batch_weather.to(device), batch_los.to(device)

        outputs = net(batch_x, adj_torch, batch_history, batch_weather, batch_los, device)
        if outputs.shape[0]!=net.mult*batch_size:
            if net.mult == 1:
                output_loc = outputs[:,:]
                output_scale = None
                output_pi = None
            elif net.mult == 2:
                batch_new = int((outputs.shape[0])/2)
                output_loc = outputs[:batch_new,:]
                output_scale = outputs[batch_new:,:]
                output_pi = None
            else:
                batch_new = int((outputs.shape[0])/3)
                output_loc = outputs[:batch_new,:]
                output_scale = outputs[batch_new:2*batch_new,:]
                output_pi = outputs[2*batch_new:,:]           
        else:
            if net.mult == 1:
                output_loc = outputs[:,:]
                output_scale = None
                output_pi = None
            elif net.mult == 2:
                output_loc = outputs[:batch_size,:]
                output_scale = outputs[batch_size:,:]
                output_pi = None
            else:
                output_loc = outputs[:batch_size,:]
                output_scale = outputs[batch_size:2*batch_size,:]
                output_pi = outputs[2*batch_size:,:]
        loss = loss_fn(output_loc, output_scale, batch_y,output_pi)

        eval_loss += loss.item()

    loss_eval.append(eval_loss/eval_num)

    ############# Save best model ###############
    if epoch == 1:
        loss_best = eval_loss
    else:
        if eval_loss < loss_best:
            loss_best = eval_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss history': loss_history,
            'average eval batch loss': eval_loss/eval_num,
            }, SAVEPATH)

    print('Ep:', epoch, '| Average loss;', running_loss/train_num, '| Evaluation loss:',eval_loss/eval_num)


