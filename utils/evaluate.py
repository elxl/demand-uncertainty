import torch
import torch.nn as nn
import numpy as np
from scipy.stats import poisson, norm, laplace, lognorm
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError


def post_process_dist(dist, loc, scale):
    
    if dist == "lognorm":
        out_predict = np.exp(loc - np.power(scale,2))
#         out_std = np.mean(np.sqrt((np.exp(scale * scale)-1)*(np.exp(2*loc+scale * scale))))
    elif dist == 'tnorm':
        out_predict = loc
#         out_std = np.mean(scale)
    elif dist == 'laplace':
        out_predict = loc
#         out_std = np.mean(scale) * np.sqrt(2)
    elif dist == 'poisson':
        out_predict = loc
#         out_std = np.sqrt(loc)
    elif dist == 'norm':
        out_predict = loc
        
        
    return out_predict
 
    
def post_process_pi(dist, loc, scale, z):
    
    if dist == "lognorm":
        s = np.exp(loc)
        shape = scale
        lognorm_dist = lognorm(s=shape, scale=s)
        lb, ub = lognorm_dist.interval(z)
        # lb, ub = lognorm.interval(z, loc, scale)
    elif dist == 'tnorm':
        lb, ub = norm.interval(z, loc, scale)
        lb = lb * (lb>0)
    elif dist == 'laplace':
        lb, ub = laplace.interval(z, loc, scale)
    elif dist == 'poisson':
        lb,ub = poisson.interval(z, loc) 
    elif dist == 'norm':
        lb, ub = norm.interval(z, loc, scale)

    return lb, ub

def eval_pi(output_lower, output_upper, target, stdout=False):
    lower = output_lower.flatten()
    lower[lower<0] = 0
    upper = output_upper.flatten()
    t = target.flatten()
    kh = (np.sign(upper-t) >= 0) * (np.sign(t-lower) >= 0)
    picp = np.mean(kh)
    mpiw = np.sum((upper-lower) * kh) / np.sum(kh)
    
    if stdout:
        print("MPIW: %.6f" %(mpiw))
        print("PICP: %.6f"%(picp))

    return mpiw, picp

def evaluate(net, loss_fn, adj_torch, dist, dataloader, z, device, batch_size):

    net.eval()
    eval_loss = 0
    eval_num = len(dataloader.dataset)
    for i,evaldata in enumerate(dataloader):
        batch_x, batch_y, batch_history, batch_weather, batch_los = evaldata

        batch_x = batch_x.float()
        batch_y = torch.squeeze(batch_y).float()
        batch_history = batch_history.float()

        batch_x, batch_y, batch_history, batch_weather, batch_los = \
            batch_x.to(device), batch_y.to(device), batch_history.to(device), \
            batch_weather.to(device), batch_los.to(device)

        outputs = net(batch_x, adj_torch, batch_history, batch_weather, batch_los, device)
        if outputs.shape[0]!=2*batch_size:
            batch_new = int((outputs.shape[0])/2)
            output_loc = outputs[:batch_new,:]
            output_scale = outputs[batch_new:,:]
        else:
            output_loc = outputs[:batch_size,:]
            output_scale = outputs[batch_size:,:]
        loss = loss_fn(output_loc, output_scale, batch_y)
        
        loss = loss_fn(output_loc, output_scale, batch_y)
        eval_loss += loss.item()

        # Get mean and variance
        if i == 0:
            test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
            test_out_var = outputs[batch_size:,:].cpu().detach().numpy()
            y_eval = batch_y.cpu().numpy()
        else:
            if outputs.shape[0]==2*batch_size:
                test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)
                test_out_var = np.concatenate((test_out_var, outputs[batch_size:,:].cpu().detach().numpy()), axis=0)
                y_eval = np.concatenate((y_eval, batch_y.cpu().numpy()), axis=0)
            else:
                batch_new = int((outputs.shape[0])/2)
                test_out_mean = np.concatenate((test_out_mean, outputs[:batch_new,:].cpu().detach().numpy()), axis=0)
                test_out_var = np.concatenate((test_out_var, outputs[batch_new:,:].cpu().detach().numpy()), axis=0)
                y_eval = np.concatenate((y_eval, batch_y.cpu().numpy()), axis=0)        
    val_out_predict = post_process_dist(dist, test_out_mean, test_out_var)
    # Point error
    mae = MeanAbsoluteError()
    val_mae = mae(torch.from_numpy(val_out_predict.flatten()), torch.from_numpy(y_eval.flatten()))
    val_mape = val_mae/np.mean(y_eval.flatten())

    lb, ub = post_process_pi(dist, test_out_mean, test_out_var, z)
    val_mpiw, val_picp = eval_pi(lb, ub, y_eval)
        
    return eval_loss/eval_num, val_mae.item(), val_mape.item(), val_mpiw, val_picp