import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader

from utils.class_loss import MVELoss
from model.class_LSTM import NN_LSTM
from utils.prepare_data import prepare_input_LSTM
from utils.evaluate import evaluate_sum

# Define hyperparameters
n_features = 2
n_time = 6
hid_g = 32
hid_fc = 32
hid_l = 32
batch_size = 32
learning_rate = 0.001
weight_decay = 0.001
dropout = 0.5
num_epochs = 20
dist = 'norm'
z = 0.95

# Define other parameters
SAVEPATH = "weights/model_sum_norm.pt"
filepath = 'data/processed/1906_sum.npz'


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Create the network
net = NN_LSTM(n_features,n_time, hid_fc, hid_g, hid_l, dist,device,dropout)
net = net.to(device)

# Define the loss function
loss_fn = MVELoss(dist)

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Read in and prepare dataset
data = np.load(filepath)
x = data['x']
y = data['y']

train_loader,val_loader,test_loader = prepare_input_LSTM(x,y,batch_size=batch_size)
train_num = len(train_loader.dataset)
eval_num = len(val_loader.dataset)
batch_number = len(train_loader)

print('Start training ...')
print(f"Training sample batches:{batch_number}")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Training loop
loss_history = []
loss_eval = []
for epoch in range(1,num_epochs+1):

    running_loss = 0.0

    # Iterate over the batches
    for i,traindata in enumerate(train_loader,1):

        net.train()

        batch_x, batch_y = traindata

        batch_x = batch_x.float()
        batch_y = torch.squeeze(batch_y).float()

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = net(batch_x)

        # Compute the loss
        if outputs.shape[0]!=2*batch_size:
            batch_new = int((outputs.shape[0])/2)
            output_loc = outputs[:batch_new]
            output_scale = outputs[batch_new:]         
        else:
            output_loc = outputs[:batch_size]
            output_scale = outputs[batch_size:]
        loss = loss_fn(output_loc, output_scale, batch_y)

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
        batch_x, batch_y = evaldata

        batch_x = batch_x.float()
        batch_y = torch.squeeze(batch_y).float()

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = net(batch_x)
        if outputs.shape[0]!=2*batch_size:
            batch_new = int((outputs.shape[0])/2)
            output_loc = outputs[:batch_new]
            output_scale = outputs[batch_new:]         
        else:
            output_loc = outputs[:batch_size]
            output_scale = outputs[batch_size:]
        loss = loss_fn(output_loc, output_scale, batch_y)

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

# Test
print('Testing ...')
loss, mae, mape, mpiw, picp = evaluate_sum(net, loss_fn, dist, test_loader, z, device, batch_size)
print('Average loss;', loss, '| MAE:',mae, '| MAPE:',mape, '| MPIW',mpiw, '| PICP:',picp)

