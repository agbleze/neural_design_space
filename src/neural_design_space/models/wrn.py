import torch
import torch.nn as nn
from torchsummary import summary
from neural_design_space.utils.utils import kernel_initializer



class WRNStem(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()
        
        self.stem_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels, kernel_size=3, 
                                                     #stride=1, 
                                                     padding=1, 
                                                     bias=False
                                                     ),
                                       nn.LazyBatchNorm2d(),
                                       nn.ReLU()
                                       )
        
    def forward(self, x):
        x = self.stem_conv(x)
        return x
    
    
    
class WRNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.LazyLinear(out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
        
        
            
class WRNIdentityBlock(nn.Module):
    def __init__(self, n_filters,
                 dropout_rate, l, stride=None):
        """n_filters: number of filters in the convolutional layers
           k: widening factor
           l: number of convolutional layers in the block
           dropout_rate: dropout rate for regularization
        """
        self.n_filters = n_filters
        #self.k = k
        self.dropout_rate = dropout_rate
        self.l = l
        
        super().__init__()
        
        if l < 2:
            raise ValueError("Number of convolutional layers in the block must be at least 2.")
  
        
        block_conv = []
        
        for _ in range(l):
            if _ == 0:
                stride = 1 #2 if not stride else stride
                conv = basic_conv(n_filters=n_filters, #k=k,
                                  stride=stride)
            else:
                conv = basic_conv(n_filters=n_filters, #k=k,
                                  stride=1)
            block_conv.append(conv)
            block_conv.append(nn.Dropout(dropout_rate))
        
        block_conv = block_conv[:-1]  # Remove the last dropout layer
        self.block_conv = nn.Sequential(*block_conv)
           
    
    def forward(self, x):
        shortcut = x
        x = self.block_conv(x)
        x += shortcut
        return x
        

def basic_conv(n_filters,#k, 
               stride):
    #out_channels = n_filters * k
    print(f"Basic convolution: {n_filters} channels")
    conv = nn.Sequential(nn.LazyBatchNorm2d(),
                        nn.ReLU(),
                        nn.LazyConv2d(out_channels=n_filters, #out_channels, 
                                        kernel_size=3, stride=stride, 
                                        padding=1, bias=False
                                        ),
                        #nn.Dropout(dropout_rate),                                
                        )
    return conv
        
        
class WRNProjectionBlock(nn.Module):
    def __init__(self, n_filters, #k, 
                 dropout_rate, l=2, stride=None):
        """
        n_filters: number of filters in the convolutional layers
        k: widening factor
        l: number of convolutional layers in the block
        dropout_rate: dropout rate for regularization
        """
        
        super().__init__()
        
        if l < 2:
            raise ValueError("Number of convolutional layers in the block must be at least 2.")
        
        self.proj = nn.Sequential(nn.LazyBatchNorm2d(),
                                #nn.ReLU(),
                                nn.LazyConv2d(out_channels=n_filters, kernel_size=1,
                                                stride=2 if not stride else stride, #stride if stride else 1, 
                                                padding=0, bias=False
                                                )
                                )        
        
        block_convs = []
        
        # conv = nn.Sequential(nn.LazyBatchNorm2d(),
        #                      nn.ReLU(),
        #                      nn.LazyConv2d(out_channels=n_filters*k, kernel_size=3)
        #                      )
        for _ in range(l):
            if _ == 0:
                stride = 2 if not stride else stride
                conv = basic_conv(n_filters=n_filters, #k=k, 
                                  stride=stride)
            else:
                conv = basic_conv(n_filters=n_filters, #k=k, 
                                  stride=1)
                
            block_convs.append(conv)
            block_convs.append(nn.Dropout(dropout_rate))
            
        block_convs = block_convs[:-1]  # Remove the last dropout layer
        self.block_conv = nn.Sequential(*block_convs)
        
        
    def forward(self, x):
        shortcut = self.proj(x)
        
        x = self.block_conv(x)
        x += shortcut
        
        return x
        

def group(out_features, #k, 
          n_blocks, dropout_rate, l, stride=None):
    """
    out_features: number of filters in the convolutional layers
    k: widening factor
    n_blocks: number of blocks in the group
    dropout_rate: dropout rate for regularization
    l: number of convolutional layers in each block
    """
    #out_channels = out_features * k
    print(f"Group with {n_blocks} blocks, each with {l} convolutional layers, and {out_features} output channels.")
    block_collection = []
    proj_conv = WRNProjectionBlock(n_filters=out_features, #k=k, 
                                   dropout_rate=dropout_rate, l=l, stride=stride)
    
    block_collection.append(proj_conv)
    for _ in range(n_blocks -1):
        block = WRNIdentityBlock(n_filters=out_features, #k=k, 
                                 dropout_rate=dropout_rate, l=l)
        block_collection.append(block)
    print(f"Number of blocks in the group: {len(block_collection)}")
    return nn.Sequential(*block_collection)


def learner(groups, depth=40):
    n_blocks = (depth -2) // 6 
    print(f"Number of blocks per group: {n_blocks}")
    learner_grps_collection = [] 
    for _, params in enumerate(groups):
        if _ == 0:
            stride = 1
            n_filters, k, dropout_rate, l = params
            n_filters = n_filters * k
            grp_blk = group(out_features=n_filters, #k=k, 
                            n_blocks=n_blocks, dropout_rate=dropout_rate, l=l, 
                            stride=stride
                            )
            learner_grps_collection.append(grp_blk)
        else:
            n_filters, k, dropout_rate, l = params
            n_filters = n_filters * k
            grp_blk = group(out_features=n_filters, #k=k,
                            n_blocks=n_blocks, dropout_rate=dropout_rate, l=l)
            learner_grps_collection.append(grp_blk)
    return nn.Sequential(*learner_grps_collection)
            
     
         
     
from typing import NamedTuple


class GroupConfig(NamedTuple):
    n_filters: int
    k: int
    dropout_rate: float
    l: int        
 


import torchvision.models as models
from torchsummary import summary


# model = models.wide_resnet50_2(weights=None).to("cuda")
# summary(model, input_size=(3, 224, 224))


if __name__ == "__main__":
    
    def kernel_initializer(m, kernel_initializer="he_normal"):
        if isinstance(m, nn.LazyConv2d) or isinstance(m, nn.LazyLinear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if kernel_initializer == "he_normal":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif kernel_initializer == "glorot_uniform":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    example_input = torch.randn(1, 3, 32, 32, device="cuda")
    # stem = WRNStem()
    # learner_module = learner(groups=group_config, depth=40)
    # classifier = WRNClassifier(num_classes=100)
    
    # model = nn.Sequential(stem, learner_module, classifier).to("cuda")
    
    
    def make_model(num_classes, group_config, depth=50):
        stem = WRNStem()
        learner_module = learner(groups=group_config, depth=depth)
        classifier = WRNClassifier(num_classes=num_classes)
        model = nn.Sequential(stem, learner_module, classifier).to("cuda")
        return model
    

    group_conv2_config = GroupConfig(n_filters=16, k=2, dropout_rate=0.3, l=2)
    group_conv3_config = GroupConfig(n_filters=32, k=2, dropout_rate=0.3, l=2)
    group_conv4_config = GroupConfig(n_filters=64, k=2, dropout_rate=0.3, l=2)
    
    
    group_config = [group_conv2_config, group_conv3_config, group_conv4_config]
    model = make_model(num_classes=100, group_config=group_config, depth=50)
    _ = model(example_input)
    model.apply(kernel_initializer)
    
    print(f"Custom WRN model summary:\n{summary(model, input_size=(3, 32, 32))}")
    
    # FLOPS at a particular conv layer can be calculated as:
    # FLOPS = 2 * H_out * W_out * C_out * C_in * K**2
    # K is the kernel size, C_out is the number of output channels,
    # C_in is the number of input channels, and 
    # H_out and W_out are the height and width of the output feature map.
    
    # for group convolution use
    # FLOPS = 2 * H_out * W_out * C_out * (C_in / G) * K**2
    # where G is the number of groups.
    
    
    from fvcore.nn import FlopCountAnalysis, get_bn_modules, flop_count_table, parameter_count_table
    from fvcore.nn.activation_count import ActivationCountAnalysis
    from torch.profiler import profile, record_function, ProfilerActivity
    flops = FlopCountAnalysis(model, example_input)
    print(f"FLOPS for the custom WRN model: {flops.total()}")
    
    print(f"Execution time profiling for the custom WRN model:")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True,
                 profile_memory=True, with_flops=True, with_modules=True,
                 with_stack=True,
                 ) as prof:
        model(example_input)
        
            
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_memory_usage",))
    
    bn_mods = get_bn_modules(model)
    #print(f"BatchNorm modules in the model: {bn_mods}")
    
    print(f" {len(bn_mods)} BatchNorm modules in the model.")
    wrn_flops = FlopCountAnalysis(model, example_input)
    wrn_act = activations=ActivationCountAnalysis(model, example_input)
    flops_table = flop_count_table(flops=wrn_flops, 
                                    activations=wrn_act, 
                                    )
    print(f"FLOPS table for the  WRN model:\n{flops_table}")
    print(f"total FLOPS for the  WRN model: {wrn_flops.total()}")
    #print(f"FLOPS by modlue for the custom WRN model:\n{wrn_flops.by_module()}")
    
    
    
    
    """
    
    From this analysis, batch Norm has low flops but high CPU total% hence would have a 
    significant impact on the overall execution time / latency of the model. Thus,
    Conv dominates the FLOPS but BatchNorm dominates the execution time / CPU overhead.
    
    This is likely because of the


    """