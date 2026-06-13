

#%%
import torch
import torch.nn as nn
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, get_bn_modules, flop_count_table, parameter_count_table
from fvcore.nn.activation_count import ActivationCountAnalysis
from torch.profiler import profile, record_function, ProfilerActivity
from typing import NamedTuple



class StemDenseNet(nn.Module):
    def __init__(self, out_channels):
        """Stem module for DenseNet, consisting of a 7x7 convolution with stride 2 and padding 3, 
            followed by batch normalization, ReLU activation, and 3x3 max pooling with stride 2 
            and padding 1 for spatial downsampling
        Args:
            out_channels (int): number of output channels for the stem convolution layer
        Returns:
            torch.Tensor: output tensor of the stem module
        """
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels=2*out_channels,
                                   kernel_size=7, stride=2,
                                   bias=False, 
                                   #padding=3
                                   ) 
        self.bn1 = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, 
                                 #padding=1
                                 )
        
        self.pad1 = nn.ZeroPad2d(padding=3)
        self.pad2 = nn.ZeroPad2d(padding=1) 
        
    def get_zeropadding(self, padding):
        return nn.ZeroPad2d(padding=padding)
    
    def forward(self, x):
        x = self.pad1(x) #self.get_zeropadding(padding=3)(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pad2(x) #self.get_zeropadding(padding=1)(x)
        x = self.pool(x)
        return x

class DenseBlockBottleneck_1x1(nn.Module):
    """Bottleneck layer for DenseNet, 1x1 convolution for dimensionality expansion

    """
    def __init__(self, out_channels, expansion_rate=4):
        """Bottleneck layer for DenseNet, 1x1 convolution for dimensionality expansion
        Args:
            out_channels (int): number of output channels for the bottleneck layer
            expansion_rate (int, optional): expansion rate for the bottleneck layer. Defaults to 4. 
                                            Use to expand the number of channels by a factor of 4 as in DenseNet-B.
                                            
        Returns:
            torch.Tensor: output tensor of the bottleneck layer
        """
        super().__init__()
        
        self.bn = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()
        self.conv = nn.LazyConv2d(out_channels=out_channels * expansion_rate, 
                                  kernel_size=1, stride=1, bias=False
                                  )
        
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x
    

class DenseBlockConv(nn.Module):
    def __init__(self, n_filters, **kwargs):
        """3x3 convolution layer for DenseNet dense block, with padding=1 to preserve feature map size
        
        Args:
            n_filters (int): number of output filters for the convolution layer
            
        Returns:
            torch.Tensor: output tensor of the convolution layer
        """
        super().__init__(**kwargs)
        
        self.bn = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()
        self.conv = nn.LazyConv2d(out_channels=n_filters, kernel_size=3, 
                                  stride=1, bias=False, padding=1
                                  )
    
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x
        
        
class DenseBlock(nn.Module):
    def __init__(self, growth_rate, bottleneck_expansion_rate=4, **kwargs):
        """Dense block for DenseNet, consisting of a bottleneck layer (optional) followed by a convolution layer, 
            with concatenation of input and output feature maps
        Args:
            growth_rate (int): growth rate for the dense block, i.e. number of filters
            bottleneck_expansion_rate (int, optional): expansion rate for the bottleneck layer. Defaults to 4. If None, no bottleneck layer is used.
        Returns:
            torch.Tensor: output tensor of the dense block
        """
        super().__init__()
        
        # BN-RE-Conv 1x1
        # dimensionality expansion, expand filters by 4 (DenseNet-B)
        # self.bn1 = nn.LazyBatchNorm2d()
        # self.act1 = nn.ReLU()
        # self.conv1 = nn.LazyConv2d(out_channels=4 * n_filters,
        #                            kernel_size=1,
        #                            stride=1,
        #                            bias=False,
        #                            )
        
        if bottleneck_expansion_rate:
            self.bottleneck = DenseBlockBottleneck_1x1(out_channels=growth_rate,
                                                        expansion_rate=bottleneck_expansion_rate
                                                        )
        # BN-RE-Conv 3x3 with padding=same to preserve same shape of feature maps
        
        self.conv = DenseBlockConv(n_filters=growth_rate)
        blocks = [self.bottleneck, self.conv] if bottleneck_expansion_rate else [self.conv]

        self.dense_block = nn.Sequential(*blocks)
        
    def forward(self, x):
        shortcut = x

        # x = self.bn1(x)
        # x = self.act1(x)
        # x = self.conv1(x)
        
        # x = self.bn2(x)
        # x = self.act2(x)
        # x = self.conv2(x)
        x = self.dense_block(x)
        x = torch.cat([shortcut, x], dim=1)
        return x
        

class TransBlock(nn.Module):
    def __init__(self, in_channels, compression=0.5, **kwargs):
        """Transition block for DenseNet, consisting of a 1x1 convolution for channel compression followed by 2x2 average pooling for spatial downsampling
        Args:          
            in_channels (int): number of input channels to the transition block
            compression (float, optional): compression factor for the transition block, i.e. the output channels will be in_channels * compression. Defaults to 0.5.
        Returns:            
                torch.Tensor: output tensor of the transition block
        """
        super().__init__()
        self.compression = compression
        n_filters = round(in_channels * self.compression)
        #print(f"Transition block: in_channels={in_channels}, compression={compression}, out_channels={n_filters}")
        #n_filters = int(int(x.shape[3]) * compression_rate)
        self.trans_block = self.get_transblock(n_filters)
        
    def get_transblock(self, n_filters):
        # BN-LI-Conv pre-activation 1x1
        self.bn1 = nn.LazyBatchNorm2d()
        self.conv1 = nn.LazyConv2d(out_channels=n_filters,
                                   kernel_size=1, stride=1,
                                   bias=False
                                   )
        self.avgpool = nn.AvgPool2d(kernel_size=2,
                                    stride=2
                                    )
        
        return nn.Sequential(self.bn1, self.conv1, self.avgpool)
        
        
    def forward(self, x):
        x = self.trans_block(x)
        return x
    
    
def group_densenet(n_blocks, growth_rate, stem_out_channels,
                   compression=0.5,
                   bottleneck_expansion_rate=4, **kwargs
                   ):
    """Construct a group of dense blocks for DenseNet, with optional transition block for downsampling and channel compression
    Args:
        n_blocks (int): number of dense blocks in the group
        growth_rate (int): growth rate for the dense blocks, i.e. number of filters added by each dense block
        stem_out_channels (int): number of output channels from the stem module, used as input channels for the first dense block in the group
        compression (float, optional): compression factor for the transition block, i.e. the output channels will be in_channels * compression. Defaults to 0.5. If None, no transition block is used.
        bottleneck_expansion_rate (int, optional): expansion rate for the bottleneck layer in the dense blocks. Defaults to 4. If None, no bottleneck layer is used.
    Returns:
        nn.Sequential: a sequential module containing the group of dense blocks and optional transition block
        int: number of output channels from the group, which can be used as input channels for the next group of dense blocks
    """
    # construct group of dense blocks
    block_collection = []
    n_current_channels = stem_out_channels
    for _ in range(n_blocks):
        dense_block = DenseBlock(growth_rate=growth_rate, 
                                 bottleneck_expansion_rate=bottleneck_expansion_rate, 
                                 stem_out_channels=stem_out_channels,
                                 **kwargs
                                 )
        block_collection.append(dense_block)
        n_current_channels += growth_rate
        
    if compression:
        trans_block = TransBlock(in_channels=n_current_channels, compression=compression)
        block_collection.append(trans_block)
        n_current_channels = int(n_current_channels * compression)
        
    
    return nn.Sequential(*block_collection), n_current_channels
        

def learner_densenet(group_configs, stem_out_channels):
    """Construct the learner module for DenseNet, consisting of multiple groups of dense blocks 
        with optional transition blocks for downsampling and channel compression
    Args:        
        group_configs (list of DenseNetGroupConfig): list of configurations for each group of dense blocks, 
        where each configuration is a DenseNetGroupConfig named tuple containing the number of blocks, 
        growth rate, compression factor, and bottleneck expansion rate for the group
        
        stem_out_channels (int): number of output channels from the stem module, 
        used as input channels for the first group of dense blocks
    Returns:        
        nn.Sequential: a sequential module containing the learner module for DenseNet, 
        which consists of multiple groups of dense blocks with optional transition blocks for 
        downsampling and channel compression
    """
    group_collection = []
    C = stem_out_channels
    for config in group_configs:
        grp, C  = group_densenet(n_blocks=config.n_blocks,
                             growth_rate=config.growth_rate,
                             compression=config.compression,
                            bottleneck_expansion_rate=config.bottleneck_expansion_rate,
                            stem_out_channels=C
                             )
        group_collection.append(grp)
    return nn.Sequential(*group_collection)

class ClassifierDenseNet(nn.Module):
    """Classifier module for DenseNet, consisting of global average pooling followed by a fully connected layer for classification
    """
    def __init__(self, num_classes):
        """Classifier module for DenseNet, consisting of global average pooling followed by a fully connected layer for classification
        Args:
            num_classes (int): number of output classes for classification
        Returns:
            torch.Tensor: output tensor of the classifier module, containing class probabilities
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.LazyLinear(out_features=num_classes)
        self.act = nn.Softmax(dim=1)
    
    
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.act(x)
        return x
        
         
# %%


class DenseNetGroupConfig(NamedTuple):
    n_blocks: int
    growth_rate: int
    compression: float
    bottleneck_expansion_rate: int

def make_model(num_classes, group_config, stem_out_channels=16, device="cuda"):
    """Construct the DenseNet model with the specified number of classes, group configurations, and stem output channels
    Args:
        num_classes (int): number of output classes for classification
        group_config (list of DenseNetGroupConfig): list of configurations for each group of dense blocks, where each configuration is a DenseNetGroupConfig named tuple containing the number of blocks, growth rate, compression factor, and bottleneck expansion rate for the group
        stem_out_channels (int, optional): number of output channels from the stem module, used as input channels for the first group of dense blocks. Defaults to 16.
        device (str, optional): device to move the model to. Defaults to "cuda".
    Returns:
        nn.Sequential: a sequential module containing the complete DenseNet model, including the stem module, learner module with multiple groups of dense blocks, and classifier module for classification
    """
    stem = StemDenseNet(out_channels=stem_out_channels)
    learner_module = learner_densenet(group_configs=group_config, 
                                      stem_out_channels=stem_out_channels
                                    )
    classifier = ClassifierDenseNet(num_classes=num_classes)
    model = nn.Sequential(stem, learner_module, classifier)
    return model.to(device)
    

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
                
                
if __name__ == "__main__":
    grp1 = DenseNetGroupConfig(n_blocks=6, growth_rate=32, compression=0.5, 
                                bottleneck_expansion_rate=4
                                )
    grp2 = DenseNetGroupConfig(n_blocks=12, growth_rate=32, compression=0.5, 
                                bottleneck_expansion_rate=4
                                )
    grp3 = DenseNetGroupConfig(n_blocks=24, growth_rate=32, compression=0.5, 
                                bottleneck_expansion_rate=4
                                )
    grp4 = DenseNetGroupConfig(n_blocks=16, growth_rate=32, compression=None, 
                                bottleneck_expansion_rate=4
                                )
    example_input = torch.randn(1, 3, 224, 224,).to("cuda")# device="cuda")
    
    group_config = [grp1, grp2, grp3, grp4]
    model = make_model(num_classes=1000, group_config=group_config, stem_out_channels=32, device="cuda")
    model.to("cuda")
    _ = model(example_input)
    model.apply(kernel_initializer)
    
    print(f"Custom DenseNet model summary:\n{summary(model, input_size=(3, 224, 224), device='cuda')}")
    
    densenet_flops = FlopCountAnalysis(model, example_input)
    densenet_act = activations=ActivationCountAnalysis(model, example_input)
    flops_table = flop_count_table(flops=densenet_flops, 
                                    activations=densenet_act, 
                                    )
    #print(f"FLOPS table for the  DenseNet model:\n{flops_table}")
    #print(f"total FLOPS for the  DenseNet model: {densenet_flops.total()}")
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #              record_shapes=True, with_flops=True, with_modules=True,
    #              with_stack=True,
    #             ) as prof:
    #     with record_function("model_inference"):
    #         _ = model(example_input)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # total FLOPS for the  DenseNet model: 2690003154 -- padding in conv, no separate padding layers
    # total FLOPS for the  DenseNet model: 2690003154 -- separate padding layers, no padding in conv layers
    
    """
    -- separate padding layers, no padding in conv layers
    ---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  Total KFLOPs  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  model_inference        40.85%      11.674ms       100.00%      28.578ms      28.578ms             1            --  
                        aten::pad         0.07%      20.000us         1.54%     441.000us     220.500us             2            --  
            aten::constant_pad_nd         0.22%      64.000us         1.47%     421.000us     210.500us             2            --  
                      aten::empty         1.71%     490.000us         1.71%     490.000us       0.814us           602            --  
                      aten::fill_         0.30%      85.000us         0.30%      85.000us      42.500us             2            --  
                     aten::narrow         0.48%     136.000us         0.50%     142.000us      17.750us             8            --  
                      aten::slice         0.06%      18.000us         0.08%      23.000us       2.875us             8            --  
                 aten::as_strided         0.03%       8.000us         0.03%       8.000us       0.889us             9            --  
                      aten::copy_         0.29%      83.000us         0.29%      83.000us      41.500us             2            --  
                     aten::conv2d         2.65%     757.000us        18.88%       5.395ms      44.958us           120   5237649.984  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 28.578ms


-- padding in conv, no separate padding layers. padding =3 and 1 in conv layers of stem module
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  Total KFLOPs  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  model_inference        39.54%      10.414ms       100.00%      26.341ms      26.341ms             1            --  
                     aten::conv2d         1.85%     488.000us        20.64%       5.438ms      45.317us           120   5237649.984  
                aten::convolution         1.68%     442.000us        19.90%       5.242ms      43.683us           120            --  
               aten::_convolution         1.52%     400.000us        17.99%       4.738ms      39.483us           120            --  
          aten::cudnn_convolution        16.70%       4.400ms        16.70%       4.400ms      36.667us           120            --  
                       aten::add_         5.85%       1.540ms         5.85%       1.540ms      12.833us           120            --  
                 aten::batch_norm         0.89%     235.000us        19.04%       5.016ms      41.800us           120            --  
     aten::_batch_norm_impl_index         2.02%     531.000us        18.52%       4.879ms      40.658us           120            --  
           aten::cudnn_batch_norm        13.92%       3.667ms        17.50%       4.609ms      38.408us           120            --  
                 aten::empty_like         1.21%     319.000us         2.26%     595.000us       4.958us           120            --  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 26.341ms

    
    
 ##################################################################################################################################
 
 -- padding in conv, no separate padding layers. padding =3 and 1 in conv layers of stem module

    FLOPS table for the  DenseNet model:
| module         | #parameters or shape   | #flops   | #activations   |
|:---------------|:-----------------------|:---------|:---------------|
| model          | 6.903M                 | 2.69G    | 6.412M         |
| 0              | 4.768K                 | 61.014M  | 0.401M         |
| 0.conv1        | 4.704K                 | 59.007M  | 0.401M         |
| 0.conv1.weight | (32, 3, 7, 7)          |          |                |
| 0.bn1          | 64                     | 2.007M   | 0              |
| 0.bn1.weight   | (32,)                  |          |                |
| 0.bn1.bias     | (32,)                  |          |                |
| 1              | 6.796M                 | 2.629G   | 6.011M         |
| 1.0            | 0.334M                 | 1.063G   | 3.337M         |
| 1.0.0          | 41.28K                 | 0.131G   | 0.502M         |
| 1.0.1          | 45.44K                 | 0.144G   | 0.502M         |
| 1.0.2          | 49.6K                  | 0.158G   | 0.502M         |
| 1.0.3          | 53.76K                 | 0.171G   | 0.502M         |
| 1.0.4          | 57.92K                 | 0.184G   | 0.502M         |
| 1.0.5          | 62.08K                 | 0.198G   | 0.502M         |
| 1.0.6          | 23.744K                | 76.569M  | 0.326M         |
| 1.1            | 1.002M                 | 0.798G   | 1.697M         |
| 1.1.0          | 50.64K                 | 40.247M  | 0.125M         |
| 1.1.1          | 54.8K                  | 43.584M  | 0.125M         |
| 1.1.2          | 58.96K                 | 46.921M  | 0.125M         |
| 1.1.3          | 63.12K                 | 50.258M  | 0.125M         |
| 1.1.4          | 67.28K                 | 53.594M  | 0.125M         |
| 1.1.5          | 71.44K                 | 56.931M  | 0.125M         |
| 1.1.6          | 75.6K                  | 60.268M  | 0.125M         |
| 1.1.7          | 79.76K                 | 63.604M  | 0.125M         |
| 1.1.8          | 83.92K                 | 66.941M  | 0.125M         |
| 1.1.9          | 88.08K                 | 70.278M  | 0.125M         |
| 1.1.10         | 92.24K                 | 73.614M  | 0.125M         |
| 1.1.11         | 96.4K                  | 76.951M  | 0.125M         |
| 1.1.12         | 0.12M                  | 95.265M  | 0.191M         |
| 1.2            | 3.314M                 | 0.661G   | 0.852M         |
| 1.2.0          | 68.84K                 | 13.711M  | 31.36K         |
| 1.2.1          | 73K                    | 14.546M  | 31.36K         |
| 1.2.2          | 77.16K                 | 15.38M   | 31.36K         |
| 1.2.3          | 81.32K                 | 16.214M  | 31.36K         |
| 1.2.4          | 85.48K                 | 17.048M  | 31.36K         |
| 1.2.5          | 89.64K                 | 17.882M  | 31.36K         |
| 1.2.6          | 93.8K                  | 18.716M  | 31.36K         |
| 1.2.7          | 97.96K                 | 19.551M  | 31.36K         |
| 1.2.8          | 0.102M                 | 20.385M  | 31.36K         |
| 1.2.9          | 0.106M                 | 21.219M  | 31.36K         |
| 1.2.10         | 0.11M                  | 22.053M  | 31.36K         |
| 1.2.11         | 0.115M                 | 22.887M  | 31.36K         |
| 1.2.12         | 0.119M                 | 23.721M  | 31.36K         |
| 1.2.13         | 0.123M                 | 24.556M  | 31.36K         |
| 1.2.14         | 0.127M                 | 25.39M   | 31.36K         |
| 1.2.15         | 0.131M                 | 26.224M  | 31.36K         |
| 1.2.16         | 0.135M                 | 27.058M  | 31.36K         |
| 1.2.17         | 0.14M                  | 27.892M  | 31.36K         |
| 1.2.18         | 0.144M                 | 28.727M  | 31.36K         |
| 1.2.19         | 0.148M                 | 29.561M  | 31.36K         |
| 1.2.20         | 0.152M                 | 30.395M  | 31.36K         |
| 1.2.21         | 0.156M                 | 31.229M  | 31.36K         |
| 1.2.22         | 0.16M                  | 32.063M  | 31.36K         |
| 1.2.23         | 0.165M                 | 32.897M  | 31.36K         |
| 1.2.24         | 0.514M                 | 0.101G   | 99.176K        |
| 1.3            | 2.146M                 | 0.107G   | 0.125M         |
| 1.3.0          | 0.103M                 | 5.135M   | 7.84K          |
| 1.3.1          | 0.107M                 | 5.344M   | 7.84K          |
| 1.3.2          | 0.111M                 | 5.552M   | 7.84K          |
| 1.3.3          | 0.115M                 | 5.761M   | 7.84K          |
| 1.3.4          | 0.12M                  | 5.969M   | 7.84K          |
| 1.3.5          | 0.124M                 | 6.178M   | 7.84K          |
| 1.3.6          | 0.128M                 | 6.387M   | 7.84K          |
| 1.3.7          | 0.132M                 | 6.595M   | 7.84K          |
| 1.3.8          | 0.136M                 | 6.804M   | 7.84K          |
| 1.3.9          | 0.14M                  | 7.012M   | 7.84K          |
| 1.3.10         | 0.144M                 | 7.221M   | 7.84K          |
| 1.3.11         | 0.149M                 | 7.429M   | 7.84K          |
| 1.3.12         | 0.153M                 | 7.638M   | 7.84K          |
| 1.3.13         | 0.157M                 | 7.846M   | 7.84K          |
| 1.3.14         | 0.161M                 | 8.055M   | 7.84K          |
| 1.3.15         | 0.165M                 | 8.263M   | 7.84K          |
| 2              | 0.102M                 | 0.152M   | 100            |
| 2.fc           | 0.102M                 | 0.102M   | 100            |
| 2.fc.weight    | (100, 1018)            |          |                |
| 2.fc.bias      | (100,)                 |          |                |
| 2.pool         |                        | 49.882K  | 0              |
total FLOPS for the  DenseNet model: 2690003154


-- separate padding layers, no padding in conv layers

| module         | #parameters or shape   | #flops   | #activations   |
|:---------------|:-----------------------|:---------|:---------------|
| model          | 6.903M                 | 2.69G    | 6.412M         |
| 0              | 4.768K                 | 61.014M  | 0.401M         |
| 0.conv1        | 4.704K                 | 59.007M  | 0.401M         |
| 0.conv1.weight | (32, 3, 7, 7)          |          |                |
| 0.bn1          | 64                     | 2.007M   | 0              |
| 0.bn1.weight   | (32,)                  |          |                |
| 0.bn1.bias     | (32,)                  |          |                |
| 1              | 6.796M                 | 2.629G   | 6.011M         |
| 1.0            | 0.334M                 | 1.063G   | 3.337M         |
| 1.0.0          | 41.28K                 | 0.131G   | 0.502M         |
| 1.0.1          | 45.44K                 | 0.144G   | 0.502M         |
| 1.0.2          | 49.6K                  | 0.158G   | 0.502M         |
| 1.0.3          | 53.76K                 | 0.171G   | 0.502M         |
| 1.0.4          | 57.92K                 | 0.184G   | 0.502M         |
| 1.0.5          | 62.08K                 | 0.198G   | 0.502M         |
| 1.0.6          | 23.744K                | 76.569M  | 0.326M         |
| 1.1            | 1.002M                 | 0.798G   | 1.697M         |
| 1.1.0          | 50.64K                 | 40.247M  | 0.125M         |
| 1.1.1          | 54.8K                  | 43.584M  | 0.125M         |
| 1.1.2          | 58.96K                 | 46.921M  | 0.125M         |
| 1.1.3          | 63.12K                 | 50.258M  | 0.125M         |
| 1.1.4          | 67.28K                 | 53.594M  | 0.125M         |
| 1.1.5          | 71.44K                 | 56.931M  | 0.125M         |
| 1.1.6          | 75.6K                  | 60.268M  | 0.125M         |
| 1.1.7          | 79.76K                 | 63.604M  | 0.125M         |
| 1.1.8          | 83.92K                 | 66.941M  | 0.125M         |
| 1.1.9          | 88.08K                 | 70.278M  | 0.125M         |
| 1.1.10         | 92.24K                 | 73.614M  | 0.125M         |
| 1.1.11         | 96.4K                  | 76.951M  | 0.125M         |
| 1.1.12         | 0.12M                  | 95.265M  | 0.191M         |
| 1.2            | 3.314M                 | 0.661G   | 0.852M         |
| 1.2.0          | 68.84K                 | 13.711M  | 31.36K         |
| 1.2.1          | 73K                    | 14.546M  | 31.36K         |
| 1.2.2          | 77.16K                 | 15.38M   | 31.36K         |
| 1.2.3          | 81.32K                 | 16.214M  | 31.36K         |
| 1.2.4          | 85.48K                 | 17.048M  | 31.36K         |
| 1.2.5          | 89.64K                 | 17.882M  | 31.36K         |
| 1.2.6          | 93.8K                  | 18.716M  | 31.36K         |
| 1.2.7          | 97.96K                 | 19.551M  | 31.36K         |
| 1.2.8          | 0.102M                 | 20.385M  | 31.36K         |
| 1.2.9          | 0.106M                 | 21.219M  | 31.36K         |
| 1.2.10         | 0.11M                  | 22.053M  | 31.36K         |
| 1.2.11         | 0.115M                 | 22.887M  | 31.36K         |
| 1.2.12         | 0.119M                 | 23.721M  | 31.36K         |
| 1.2.13         | 0.123M                 | 24.556M  | 31.36K         |
| 1.2.14         | 0.127M                 | 25.39M   | 31.36K         |
| 1.2.15         | 0.131M                 | 26.224M  | 31.36K         |
| 1.2.16         | 0.135M                 | 27.058M  | 31.36K         |
| 1.2.17         | 0.14M                  | 27.892M  | 31.36K         |
| 1.2.18         | 0.144M                 | 28.727M  | 31.36K         |
| 1.2.19         | 0.148M                 | 29.561M  | 31.36K         |
| 1.2.20         | 0.152M                 | 30.395M  | 31.36K         |
| 1.2.21         | 0.156M                 | 31.229M  | 31.36K         |
| 1.2.22         | 0.16M                  | 32.063M  | 31.36K         |
| 1.2.23         | 0.165M                 | 32.897M  | 31.36K         |
| 1.2.24         | 0.514M                 | 0.101G   | 99.176K        |
| 1.3            | 2.146M                 | 0.107G   | 0.125M         |
| 1.3.0          | 0.103M                 | 5.135M   | 7.84K          |
| 1.3.1          | 0.107M                 | 5.344M   | 7.84K          |
| 1.3.2          | 0.111M                 | 5.552M   | 7.84K          |
| 1.3.3          | 0.115M                 | 5.761M   | 7.84K          |
| 1.3.4          | 0.12M                  | 5.969M   | 7.84K          |
| 1.3.5          | 0.124M                 | 6.178M   | 7.84K          |
| 1.3.6          | 0.128M                 | 6.387M   | 7.84K          |
| 1.3.7          | 0.132M                 | 6.595M   | 7.84K          |
| 1.3.8          | 0.136M                 | 6.804M   | 7.84K          |
| 1.3.9          | 0.14M                  | 7.012M   | 7.84K          |
| 1.3.10         | 0.144M                 | 7.221M   | 7.84K          |
| 1.3.11         | 0.149M                 | 7.429M   | 7.84K          |
| 1.3.12         | 0.153M                 | 7.638M   | 7.84K          |
| 1.3.13         | 0.157M                 | 7.846M   | 7.84K          |
| 1.3.14         | 0.161M                 | 8.055M   | 7.84K          |
| 1.3.15         | 0.165M                 | 8.263M   | 7.84K          |
| 2              | 0.102M                 | 0.152M   | 100            |
| 2.fc           | 0.102M                 | 0.102M   | 100            |
| 2.fc.weight    | (100, 1018)            |          |                |
| 2.fc.bias      | (100,)                 |          |                |
| 2.pool         |                        | 49.882K  | 0              |
total FLOPS for the  DenseNet model: 2690003154
    
    
    """