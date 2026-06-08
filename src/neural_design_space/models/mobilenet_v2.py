import torch
import torch.nn as nn
from torchinfo import summary
from typing import NamedTuple, List
from copy import deepcopy
from neural_design_space.utils.utils import kernel_initializer

class LazyDepthwiseConv2d(nn.LazyConv2d):
    
    def initialize_parameters(self, input):
        
        x = input[0] if isinstance(input, (tuple, list)) else input
        device = x.device
        dtype = x.dtype
        
        in_ch = int(x.shape[1])
        
        out_ch = getattr(self, "out_channels", None)
        
        if out_ch is None or out_ch == 0:
            out_ch = in_ch
        
        self.groups = in_ch
        self.out_channels = in_ch
        self.in_channels = in_ch
        
        k = self.kernel_size
        if isinstance(k, int):
            k = (k, k)
            
                
        if self.in_channels != self.out_channels:
            raise ValueError(f"Depthwise conv expects in_channels == out_channels but got in_channels {self.in_channels} != {self.in_channels}")
        
        
        weight_shapee = (self.out_channels, self.in_channels // self.groups, *k)
        
        self.weight = nn.Parameter(torch.empty(weight_shapee,device=device, dtype=dtype))
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype))
        else:
            self.bias = None
        
  

class MobileNetV2Stem(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.LazyConv2d(out_channels=32, kernel_size=3,
                                  padding=1,
                                  bias=False,
                                  stride=2
                                  )
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu6_1 = nn.ReLU6()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu6_1(x)
        return x
        

class MobileNetResidualBlockV2(nn.Module):
    def __init__(self, out_channels, width_multiplier, expansion_rate):
        super().__init__()
        
        out_channels = int(width_multiplier * out_channels)
        expanded_channels = int(out_channels * expansion_rate) if expansion_rate else out_channels
        self.conv1x1 = nn.LazyConv2d(out_channels=expanded_channels,
                                     kernel_size=1,
                                     bias=False,
                                     stride=1
                                     )
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu6_1 = nn.ReLU6()
        self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None,
                                                  kernel_size=3,
                                                  bias=False,
                                                  padding=1,
                                                  stride=1,
                                                  )
        self.bn2 = nn.LazyBatchNorm2d()
        self.relu6_2 = nn.ReLU6()
        
        self.pointwise_conv = nn.LazyConv2d(out_channels=out_channels,
                                            kernel_size=1,
                                            bias=False
                                            )
        self.bn3 = nn.LazyBatchNorm2d()
        
    def forward(self, x):
        shortcut = x
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = self.relu6_1(x)
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.relu6_2(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x += shortcut
        return x
    
    
class MobileNetNonResidualBlockV2(nn.Module):
    def __init__(self, out_channels, width_multiplier, expansion_rate, stride=None):
        super().__init__()
        out_channels = int(out_channels * width_multiplier)
        
        expanded_channels = int(out_channels * expansion_rate) if expansion_rate else out_channels
        
        self.expansion_conv = nn.LazyConv2d(out_channels=expanded_channels,
                                            kernel_size=1,
                                            bias=False,
                                            )
        self.bn1 = nn.LazyBatchNorm2d()
        self.relu6_1 =nn.ReLU6()
        
        self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None,
                                                  kernel_size=3,
                                                  stride=2 if not stride else stride,
                                                  padding=1,
                                                  bias=False
                                                  )
        self.bn2 = nn.LazyBatchNorm2d()
        self.relu6_2 = nn.ReLU6()
        
        self.pointwise_conv = nn.LazyConv2d(out_channels=out_channels,
                                            kernel_size=1,
                                            bias=False
                                            )
        self.bn3 = nn.LazyBatchNorm2d()
        
    def forward(self, x):
        x = self.expansion_conv(x)
        x = self.bn1(x)
        x = self.relu6_1(x)
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.relu6_2(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        return x
        

class MobileNetV2LastLearnerBlock(nn.Module):
    def __init__(self, out_channels, **kwargs
                 ):
        super().__init__()
        self.last_learner_conv = nn.LazyConv2d(kernel_size=1,
                                               out_channels=out_channels,
                                               bias=False
                                               )
        self.bn1 = nn.LazyBatchNorm2d()

    def forward(self, x):
        x = self.last_learner_conv(x)
        x = self.bn1(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1x1 = nn.LazyConv2d(out_channels=num_classes,
                                     kernel_size=1
                                     )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1x1(x)
        x = self.softmax(x)
        return x

def group(*, num_blocks, out_channels, width_multiplier, expansion_rate, 
          first_layer_stride,
          **kwargs
          ):
    
    grp_blks = []
    for i in range(num_blocks):
        if num_blocks == 1 and i == 0:
            blk = MobileNetNonResidualBlockV2(out_channels=out_channels,
                                               width_multiplier=width_multiplier,
                                               expansion_rate=expansion_rate,
                                               stride=first_layer_stride, #1
                                               )
            
            
        elif i == 0 and num_blocks > 1:    
            blk = MobileNetNonResidualBlockV2(out_channels=out_channels,
                                               width_multiplier=width_multiplier,
                                               expansion_rate=expansion_rate,
                                               stride=first_layer_stride #2
                                               )
        else:    
            blk = MobileNetResidualBlockV2(out_channels=out_channels,
                                      width_multiplier=width_multiplier,
                                      expansion_rate=expansion_rate
                                      )
        grp_blks.append(blk)
        
    return nn.Sequential(*grp_blks)


class BlockConfig(NamedTuple):
    out_channels: int
    num_blocks: int 
    first_layer_stride: int       
    
class MobileNetV2GroupConfig(NamedTuple):
    width_multiplier: float
    expansion_rate: int
    block_config: List[BlockConfig]
    
    
    
def learner(configs: MobileNetV2GroupConfig):
    learner_groups = []
    last_grp_idx = len(configs.block_config) -1
    
    for idx, conf in enumerate(configs.block_config):
        if idx == last_grp_idx:
            grp = MobileNetV2LastLearnerBlock(**configs._asdict(), **conf._asdict())
            
        _config = deepcopy(configs)._asdict()
        if idx == 0:
            _config["expansion_rate"] = 1
        
            grp = group(**_config, **conf._asdict())
            
        elif idx != last_grp_idx:
            grp = group(**_config, **conf._asdict())
        
        learner_groups.append(grp)
            
    return nn.Sequential(*learner_groups)
    

def make_model(num_classes, learner_config, device="cuda"):
    stem = MobileNetV2Stem()
    learner_module = learner(learner_config)
    classifier = Classifier(num_classes=num_classes)
    model = nn.Sequential(stem, learner_module, classifier)
    model.to(device)
    return model
    
    
if __name__ == "__main__":
    block_config = [BlockConfig(out_channels=16, num_blocks=1, first_layer_stride=1),
                    BlockConfig(out_channels=24, num_blocks=2, first_layer_stride=2),
                    BlockConfig(out_channels=32, num_blocks=3, first_layer_stride=2),
                    BlockConfig(out_channels=64, num_blocks=4, first_layer_stride=2),
                    BlockConfig(out_channels=96, num_blocks=3, first_layer_stride=1),
                    BlockConfig(out_channels=160, num_blocks=3, first_layer_stride=2),
                    BlockConfig(out_channels=320, num_blocks=1, first_layer_stride=1),
                    BlockConfig(out_channels=1280, num_blocks=1, first_layer_stride=1)
                    ]
    
    
    learner_config = MobileNetV2GroupConfig(width_multiplier=1,
                                            expansion_rate=6,
                                            block_config=block_config
                                            )
    
    device = "cuda"
    data = torch.randn((1, 3, 224, 224)).to(device)
    model = make_model(num_classes=1000, learner_config=learner_config,
                       device=device
                       )
    _ = model(data)
    model.apply(kernel_initializer)
    
    summary(model=model, input_data=data,
            col_names=["input_size", "output_size", "num_params",
                       "mult_adds",
                       ],
            depth=4
            )
    
    
    
    
