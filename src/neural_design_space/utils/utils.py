import torch
import torch.nn as nn
             
           
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
       