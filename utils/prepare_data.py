import torch
import libpysal
import scipy
import numpy as np
import pandas as pd
import pickle as pkl
from torch.utils.data import DataLoader
from class_dataset import NY_Data


def prepare_for_torch(device, train_extent, data, adj, spatial, downtown_filter, adj_type, val=True, bootstrap=False):

    if len(data['x']) == 2:
        # there is no validation set
        [x_train,x_test] = data['x']
        [y_train,y_test] = data['y'] 
        [ref_train, ref_test] = data['ref'] 
        [los_train, los_test] = data['los']
        [weather_train, weather_test] = data['weather'] 
        [qod_train, qod_test] = data['qod']
    elif val == False:
        # there is a validation set but we are not using it
        [x_train,_,x_test] = data['x']
        [y_train,_,y_test] = data['y'] 
        [ref_train,_, ref_test] = data['ref'] 
        [los_train,_, los_test] = data['los']
        [weather_train,_, weather_test] = data['weather'] 
        [qod_train,_, qod_test] = data['qod']
    elif len(data['x'])==3:
        [x_train,x_val,x_test] = data['x']
        [y_train,y_val,y_test] = data['y'] 
        [ref_train,ref_val,ref_test] = data['ref'] 
        [los_train,los_val,los_test] = data['los']
        [weather_train,weather_val,weather_test] = data['weather'] 
        [qod_train,qod_val,qod_test] = data['qod']

   
    if train_extent == 'all':
        
        trainset = NY_Data(torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(ref_train), 
                torch.Tensor(weather_train), torch.Tensor(los_train), torch.LongTensor(qod_train))
        if bootstrap:
            sampler = torch.utils.data.RandomSampler(trainset, replacement=True)
            trainloader = DataLoader(trainset, batch_size = 16, num_workers=10, sampler=sampler)
        else:
            trainloader = DataLoader(trainset, batch_size = 16, shuffle=True, num_workers=10)
        trainloader_test = DataLoader(trainset, batch_size = 8, shuffle=False, num_workers=10)
        
        if len(data['x']) == 3 and val == True:
            valset = NY_Data(torch.Tensor(x_val), torch.Tensor(y_val), torch.Tensor(ref_val), 
                    torch.Tensor(weather_val), torch.Tensor(los_val), torch.LongTensor(qod_val))
            valloader = DataLoader(valset, batch_size = 8, shuffle=False, num_workers=10)
            y_val_eval = y_val

        testset = NY_Data(torch.Tensor(x_test), torch.Tensor(y_test), torch.Tensor(ref_test), 
                torch.Tensor(weather_test), torch.Tensor(los_test), torch.LongTensor(qod_test))
        testloader = DataLoader(testset, batch_size = 8, shuffle=False, num_workers=10)

        adj_torch = torch.tensor([])
        for t in adj_type:
            if t in adj.keys():
                adj_torch = torch.cat((adj_torch, torch.Tensor(adj[t][:,:,np.newaxis])),dim=2)
            else:
#                 print(t, "not available. Skipped...")
                pass
        adj_torch = adj_torch.to(device)
        spatial_torch = torch.Tensor(spatial).to(device)

        y_train_eval = y_train
        y_test_eval = y_test

    elif train_extent == 'downtown':
        n_stations = np.sum(downtown_filter)
        trainset = NY_Data(torch.Tensor(x_train[:,:,downtown_filter,:]), torch.Tensor(y_train[:,:,downtown_filter]), 
                torch.Tensor(ref_train[:,:,downtown_filter]), 
                torch.Tensor(weather_train), torch.Tensor(los_train[:,:,downtown_filter]), 
                torch.LongTensor(qod_train))
        if bootstrap:
            sampler = torch.utils.data.RandomSampler(trainset, replacement=True)
            trainloader = DataLoader(trainset, batch_size = 64, num_workers=10, sampler=sampler)
        else:
            trainloader = DataLoader(trainset, batch_size = 64, shuffle=True, num_workers=10)
        trainloader_test = DataLoader(trainset, batch_size = 64, shuffle=False, num_workers=10)

        if len(data['x']) == 3 and val == True:
            valset = NY_Data(torch.Tensor(x_val[:,:,downtown_filter,:]), torch.Tensor(y_val[:,:,downtown_filter]), 
                    torch.Tensor(ref_val[:,:,downtown_filter]), 
                    torch.Tensor(weather_val), torch.Tensor(los_val[:,:,downtown_filter]), 
                    torch.LongTensor(qod_val))
            valloader = DataLoader(valset, batch_size = len(y_val), shuffle=False, num_workers=10)
            y_val_eval = y_val[:,:,downtown_filter]

        testset = NY_Data(torch.Tensor(x_test[:,:,downtown_filter,:]), torch.Tensor(y_test[:,:,downtown_filter]), 
                torch.Tensor(ref_test[:,:,downtown_filter]), 
                torch.Tensor(weather_test), torch.Tensor(los_test[:,:,downtown_filter]), 
                torch.LongTensor(qod_test))
        testloader = DataLoader(testset, batch_size = len(y_test), shuffle=False, num_workers=10)

        adj_torch = torch.tensor([])
        for t in adj_type:
            adj_torch = torch.cat((adj_torch, torch.Tensor(adj[t][downtown_filter,:][:,downtown_filter][:,:,np.newaxis])),dim=2)
        adj_torch = adj_torch.to(device)
        spatial_torch = torch.Tensor(spatial[downtown_filter]).to(device)

        y_train_eval = y_train[:,:,downtown_filter]
        y_test_eval = y_test[:,:,downtown_filter]

    else:
        print("Error")
        return

    if len(data['x']) == 2 or val == False:
        return  trainloader, trainloader_test, testloader, adj_torch, spatial_torch, y_train_eval, y_test_eval
    else:
        return  trainloader, trainloader_test, valloader, testloader, adj_torch, spatial_torch, y_train_eval, y_val_eval, y_test_eval
    
def prepare_input(x, y, adj, history, weather, los, train=0.6, val=0.2, test=0.2):
    
    dataset = NY_Data(x, y, adj, history, weather, los)
    num_samples = len(x)
    num_train = int(train * num_samples)
    num_val = int(val * num_samples)
    num_test = num_samples - num_train - num_val

    # Create subsets for train, validation, and test
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(42)
    )

    # Create separate DataLoaders for each set
    batch_size = 32  # Adjust this according to your requirements

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader