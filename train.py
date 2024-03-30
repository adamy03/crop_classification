import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from models import UNet
from torch.utils.data import DataLoader, random_split
from dice_loss import dice_coeff

def train_model(model, 
          optimizer, 
          loss_fn, 
          device, 
          epochs, 
          train_loader, 
          valid_loader):
    
    train_loss = []
    train_dice = []
    vaild_loss = []
    valid_dice = []
    bar = tqdm(range(epochs), position=0)
    
    for epoch in bar:
        model.train()
        train_loss_e = []
        train_dice_e = []
        
        for idx, batch in enumerate(train_loader):
            samples, labels = batch
            samples, labels = samples.to(device), labels.to(device)
            
            preds = pred_step(samples, model)
            loss, dice = eval_step(preds, labels, loss_fn, True)
            loss.backward()
            optimizer.step()
            
            train_loss_e.append(loss.item())
            train_dice_e.append(dice)
            
        model.eval()
        for idx, batch in enumerate(valid_loader):
            with torch.no_grad():
                samples, labels = batch
                samples, labels = samples.to(device), labels.to(device)
                
                preds = pred_step(samples, model)
                loss, dice = eval_step(preds, labels, loss_fn, True)
                
                vaild_loss.append(loss.item())
                valid_dice.append(dice)
              
        train_loss.append(np.mean(train_loss_e))
        train_dice.append(np.mean(train_dice_e))      
        
    return model, {
        'train_loss': train_loss,
        'valid_loss': vaild_loss,
        'train_dice': train_dice,
        'valid_dice': valid_dice
        }      
        

def pred_step(samples, model):
    preds = model.forward(samples)
    return preds

def eval_step(preds, labels, loss_fn, compute_dice=False):
    loss = loss_fn(preds, labels)
    
    if compute_dice:
        dice = dice_coeff(preds, labels)
        return loss, dice
    
    return loss