import torch
import torch.nn as nn
from copy import deepcopy
            
           
def kernel_initializer(m, kernel_initializer="he_normal"):
    """Kernel initializer to be used on initializing Lazy modules, using He normal initialization for convolutional layers and Xavier uniform initialization for linear layers
    Args:  m (nn.Module): module to initialize
        kernel_initializer (str, optional): type of kernel initializer to use. Defaults to 'he_normal'. Options are 'he_normal' and 'glorot_uniform'
    Returns:    
        None
    """
    if isinstance(m, nn.LazyConv2d) or isinstance(m, nn.LazyLinear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
            

def train_batch(input, model, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad()
    noisy_input, clean_input = input
    noisy_input = noisy_input.to(device)
    clean_input = clean_input.to(device)
    output = model(noisy_input)
    loss = criterion(output, clean_input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(input, model, criterion, device):
    model.eval().to(device)
    noisy_input, clean_input = input
    noisy_input = noisy_input.to(device)
    clean_input = clean_input.to(device)
    output = model(noisy_input)
    loss = criterion(output, clean_input)
    return loss




def train_convaec(model, criterion, optimizer, trn_dl, val_dl, num_epochs, scheduler):
    model.to(device)
    criterion.to(device)
    log = Report(num_epochs)
    history = {"train_epoch_loss": [], "val_epoch_loss": []}
    best_model = None
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        N = len(trn_dl)
        running_train_loss = 0.0
        model.train()
        for ix, data in enumerate(trn_dl):
            loss = train_batch(data, model, criterion, optimizer)
            running_train_loss += loss
            log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end="\r")
        train_epoch_loss = running_train_loss / N    
        history["train_epoch_loss"].append(train_epoch_loss)
        
        N = len(val_dl)
        running_val_loss = 0.0
        model.eval()
        for ix, data in enumerate(val_dl):
            loss = validate_batch(data, model, criterion)
            running_val_loss += loss
            log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end="\r")
        val_epoch_loss = running_val_loss / N
        history["val_epoch_loss"].append(val_epoch_loss)
        log.report_avgs(epoch+1)
        
        if scheduler:
            if not torch.isnan(val_epoch_loss):
                scheduler.step(val_epoch_loss)    
                    
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = deepcopy(model)
        
    log.plot_epochs(log=True)
    return best_model, log, history