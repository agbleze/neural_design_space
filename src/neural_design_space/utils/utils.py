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