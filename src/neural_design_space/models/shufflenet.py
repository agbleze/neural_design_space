import torch
import torch.nn as nn
from torchinfo import summary
from typing import List, NamedTuple
from neural_design_space.utils.utils import kernel_initializer


class ShuffleNetStem(nn.Module):
    def __init__(self, out_channels=24, **kwargs):
        super().__init__()
        
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, stride=2,
                                  bias=False
                                  )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x    
    
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups        
        x = x.view(batch_size, self.groups, channels_per_group, height, width)        
        x = x.transpose(1, 2).contiguous()        
        x = x.view(batch_size, num_channels, height, width)        
        return x
    
    
class LazyDepthwiseConv2d(nn.LazyConv2d):
    def initialize_parameters(self, input):
        x = input[0] if isinstance(input, (list, tuple)) else input
        device = x.device
        dtype = x.dtype
        self.in_channels = int(x.shape[1])
        
        if not self.out_channels:
            self.out_channels = self.in_channels
            
        #if self.groups == 0:
        self.groups = self.in_channels
        
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
            
        weight_shape = (self.out_channels, self.in_channels // self.groups, *kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape, device=device, dtype=dtype))
        
        if self.bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype))
        

class LazyGroupPointwiseRealignBottleneck(nn.LazyConv2d):
    
    def __init__(self, out_channels, kernel_size, 
                 groups, bias=False,
                 stem_out_channels=None, 
                 bottleneck_ratio=0.25, 
                 **kwargs
                 ):
        super().__init__(out_channels, kernel_size, bias=bias, groups=groups)
        self.stem_out_channels = stem_out_channels
        self.bottleneck_ratio = bottleneck_ratio
        
    def initialize_parameters(self, input):
        x = input[0] if isinstance(input, (list, tuple)) else input
        device = x.device
        dtype = x.dtype
        self.in_channels = int(x.shape[1])
        out_ch = self.out_channels
        intermediate_channels = int(self.in_channels // self.bottleneck_ratio) / 2
        self.out_channels = int(out_ch - self.stem_out_channels) if self.stem_out_channels else int(out_ch - intermediate_channels)
        
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        weight_shape = (self.out_channels, self.in_channels // self.groups, *kernel_size)
        
        self.weight = nn.Parameter(torch.empty(weight_shape, device=device, dtype=dtype))
        if self.bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype))

class ShuffleNetResidualBlock(nn.Module):
    def __init__(self, out_channels, groups, width_multiplier=1, **kwargs):
        super().__init__()
        self.groups = groups
        out_channels = int(out_channels * width_multiplier)
        bottleneck_channels = int(out_channels * 0.25)
        
        self.pointwise_group_conv1 = nn.LazyConv2d(out_channels=bottleneck_channels, 
                                                    kernel_size=1, 
                                                    bias=False,
                                                    groups=groups,
                                                    )
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu1 = nn.ReLU(inplace=True)
        
        self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None,
                                                  kernel_size=3, 
                                                  stride=1, 
                                                  padding=1, 
                                                  bias=False,
                                                  )
        self.bn2 = nn.LazyBatchNorm2d()
        
        self.pointwise_conv2 = nn.LazyConv2d(out_channels=out_channels, 
                                            kernel_size=1, 
                                            bias=False, 
                                            groups=groups
                                            )
        self.bn3 = nn.LazyBatchNorm2d()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x        
        out = self.pointwise_group_conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = ChannelShuffle(groups=self.groups)(out)
        out = self.depthwise_conv(out)
        out = self.bn2(out)
        out = self.pointwise_conv2(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu2(out)
        return out

        
class ShuffleNetDenseBlock(nn.Module):
    def __init__(self, out_channels, groups, width_multiplier=1, 
                 bottleneck_ratio=0.25, 
                 **kwargs
                 ):
        super().__init__()
        self.groups = groups
        self.bottleneck_ratio = bottleneck_ratio
        out_channels = int(out_channels * width_multiplier)
        
        bottleneck_channels = int(out_channels * self.bottleneck_ratio)
        
        self.pointwise_group_conv1 = nn.LazyConv2d(out_channels=bottleneck_channels,
                                                    kernel_size=1, 
                                                    bias=False,
                                                    groups=groups,
                                                    )
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu1 = nn.ReLU(inplace=True)
        
        self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None,
                                                  kernel_size=3, 
                                                  stride=2, 
                                                  padding=1, 
                                                  bias=False
                                                  )
        self.bn2 = nn.LazyBatchNorm2d()
        
        self.pointwise_group_conv2 = LazyGroupPointwiseRealignBottleneck(out_channels=out_channels,
                                                                        kernel_size=1,
                                                                        groups=groups,
                                                                        bias=False,
                                                                        bottleneck_ratio=self.bottleneck_ratio,
                                                                        **kwargs
                                                                        )
        self.bn3 = nn.LazyBatchNorm2d()
        
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        pooled_x = self.downsample(x)
        x = self.pointwise_group_conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = ChannelShuffle(groups=self.groups)(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_group_conv2(x)
        x = self.bn3(x)
        
        print(f"out shape before concat: {x.shape}, shortcut shape: {pooled_x.shape}")
        x = torch.concat([x, pooled_x], dim=1)
        x = self.relu2(x)
        return x        

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(out_features=num_classes, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        x = self.softmax(x)
        return x   


class BlockConfig(NamedTuple):
    out_channels: int
    num_blocks: int
    stem_out_channels: int = None
          

class GroupConfig(NamedTuple):
    block_config: List[BlockConfig]
    width_multiplier: int = 1
    groups: int = 3
    
    bottleneck_ratio: float = 0.25


def create_group_blocks(*, out_channels, num_blocks, groups, 
                        width_multiplier,
                        **kwargs
                        ):
    blocks = []
    for i in range(num_blocks):
        if i == 0:
            blocks.append(ShuffleNetDenseBlock(out_channels=out_channels, 
                                              groups=groups, 
                                              width_multiplier=width_multiplier,
                                              **kwargs
                                              )
                          )
        else:
            blocks.append(ShuffleNetResidualBlock(out_channels=out_channels, 
                                                groups=groups, 
                                                width_multiplier=width_multiplier,
                                                **kwargs
                                                )
                          )
    return nn.Sequential(*blocks)


def create_learner(*, configs: GroupConfig, **kwargs):
    blocks = []
    for block_config in configs.block_config:
        blk = create_group_blocks(**block_config._asdict(),
                                  **configs._asdict(),
                                  **kwargs
                                  )
        blocks.append(blk)
    return nn.Sequential(*blocks)


def make_model(num_classes, learner_configs, data, device, initializer_type="he_normal"):
    stem = ShuffleNetStem(out_channels=24)
    learner_module = create_learner(configs=learner_configs)
    classifier = Classifier(num_classes=num_classes)
    data = data.to(device)
    model = nn.Sequential(stem, learner_module, classifier).to(device)
    _ = model(data)
    model.apply(lambda m: kernel_initializer(m, initializer_type=initializer_type))
    model = model.to(device)
    return model


if __name__ == "__main__":
    data = torch.randn(1, 3, 224, 224)
    block_config = [BlockConfig(out_channels=240, num_blocks=4, stem_out_channels=24),
                    BlockConfig(out_channels=480, num_blocks=8),
                    BlockConfig(out_channels=960, num_blocks=4)
                    ]
    group_config = GroupConfig(width_multiplier=1, groups=3, block_config=block_config)
    
    model = make_model(num_classes=1000, learner_configs=group_config, 
                       data=data, device="cpu", initializer_type="he_normal")
    summary(model, input_size=(1, 3, 224, 224))
    