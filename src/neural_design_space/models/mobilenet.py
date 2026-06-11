import torch
import torch.nn as nn
from typing import NamedTuple, List
from torchinfo import summary 
from neural_design_space.utils.utils import kernel_initializer


class LazyDepthwiseConv2d(nn.LazyConv2d):
    def initialize_parameters(self, input):
        # Accept either a Tensor or a tuple/list as provided by lazy hooks
        x = input[0] if isinstance(input, (list, tuple)) else input
        device = x.device
        dtype = x.dtype

        # infer in_channels from input tensor
        in_ch = int(x.shape[1])

        # If out_channels was not provided, default to in_channels for depthwise
        out_ch = getattr(self, "out_channels", None)
        if out_ch is None or out_ch == 0:
            out_ch = in_ch

        k = self.kernel_size
        if isinstance(k, int):
            k = (k, k)

        self.in_channels = int(in_ch)
        self.out_channels = int(out_ch)
        self.groups = int(in_ch)  # depthwise conv: groups == in_channels

        # depthwise constraint
        if self.out_channels != self.in_channels:
            raise ValueError(
                f"Depthwise conv requires out_channels == in_channels, "
                f"got out={self.out_channels}, in={self.in_channels}"
            )

        # Weight shape for Conv2d
        weight_shape = (self.out_channels, self.in_channels // self.groups, *k)

        # Create parameters on correct device/dtype
        self.weight = nn.Parameter(torch.empty(weight_shape, device=device, dtype=dtype))
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype))
        else:
            self.bias = None

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, out_channels, stride, width_multiplier=1):
        super().__init__()
        
        out_channels = int(out_channels * width_multiplier)
        
        self.depthwise_conv = nn.Sequential(LazyDepthwiseConv2d(out_channels=None,
                                                                kernel_size=3,
                                                                stride=stride,
                                                                padding=1
                                                                ),
                                            nn.LazyBatchNorm2d(),
                                            nn.ReLU6(),
                                            )
        self.pointwise_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels,
                                                          kernel_size=1,
                                                          stride=1
                                                          ),
                                            nn.LazyBatchNorm2d(),
                                            nn.ReLU6()
                                            )
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class MobileNetStem(nn.Module):
    def __init__(self, width_multiplier=1):
        super().__init__()
        self.zeropad = nn.ZeroPad2d(1)
        self.conv1 = nn.LazyConv2d(out_channels=32, kernel_size=3,
                                   stride=2, padding=1,
                                   )
        self.depthwise_separable_conv = DepthwiseSeparableConv(out_channels=64, stride=1,
                                                               width_multiplier=width_multiplier
                                                               )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_separable_conv(x)
        return x
        

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.avg_globalpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv1x1 = nn.LazyConv2d(kernel_size=1,
                                     out_channels=num_classes
                                     )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.avg_globalpool(x)
        x = self.conv1x1(x)
        x = self.softmax(x)
        return x

def group(*, num_blocks, out_channels, width_multiplier, 
          stride=None,
          **kwargs
          ):
    group_blocks = []
    
    for i in range(num_blocks):
        depthwise_block = DepthwiseSeparableConv(out_channels=out_channels,
                                                 width_multiplier=width_multiplier,
                                                 stride=2 if i == 0 else 1
                                                 )    
        group_blocks.append(depthwise_block) 
    return nn.Sequential(*group_blocks)


def learner(configs, **kwargs):
    learner_groups = []
    
    last_gro_index = len(configs.block_config) - 1
    
    
    for idx, grp_config in enumerate(configs.block_config):
        group_block = group(**configs._asdict(), **grp_config._asdict())
        learner_groups.append(group_block)
        
    return nn.Sequential(*learner_groups)
    
    
def make_model(num_classes, learner_configs, device="cuda"):
    stem = MobileNetStem(width_multiplier=learner_configs.width_multiplier) 
    learner_module = learner(configs=learner_configs)  
    classifier = Classifier(num_classes=num_classes)
    
    model = nn.Sequential(stem, learner_module, classifier)
    model.to(device) 
    return model

        
class MobileNetBlockConfig(NamedTuple):
    out_channels: int
    num_blocks: int
    

class MobileNetGroupConfig(NamedTuple):
    width_multiplier: float
    block_config: List[MobileNetBlockConfig]  
    
    
    
block_config = [MobileNetBlockConfig(out_channels=128, num_blocks=2),
                MobileNetBlockConfig(out_channels=256, num_blocks=2),
                MobileNetBlockConfig(out_channels=512, num_blocks=6),
                MobileNetBlockConfig(out_channels=1024, num_blocks=2)
                ]     



group_configs = MobileNetGroupConfig(width_multiplier=1,
                                     block_config=block_config
                                     )


if __name__ == "__main__":
    device = "cuda"
    data = torch.randn(10, 3, 224, 224).to(device)
    model = make_model(num_classes=1000, learner_configs=group_configs, device=device)
    _ = model(data)
    model.apply(kernel_initializer)
    
    summary(model=model, input_data=data, device='cuda',
            verbose=1,
            mode="train",
            col_names=["input_size", "output_size", "num_params",
                       "mult_adds",
                       ],
            depth=4,
            ) 
    
