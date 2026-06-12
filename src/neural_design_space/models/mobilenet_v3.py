import torch
import torch.nn as nn
from torchinfo import summary
from typing import NamedTuple, List, Literal
from neural_design_space.utils.utils import kernel_initializer


NON_LINEARITY_VALUES = ["relu", "hswish"]
KERNEL_SIZE_VALUES = ["3x3", "5x5"]


class LazyDepthwiseConv2d(nn.LazyConv2d):
    def initialize_parameters(self, input):
        x = input[0] if isinstance(input, (tuple, list)) else input
        device = x.device
        dtype = x.dtype
        in_ch = x.shape[1]
        if not self.out_channels:
            self.out_channels = int(in_ch)
            
        self.in_channels = int(in_ch)
        self.groups = int(in_ch)
        
        if isinstance(self.kernel_size, int):
            k = (self.kernel_size, self.kernel_size)
        else:
            k = self.kernel_size
            
        weight_shape = (self.out_channels, self.in_channels // self.groups, *k)
        self.weight = nn.Parameter(torch.empty(weight_shape, dtype=dtype, device=device))
        
        if self.bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, device=device, dtype=dtype))
        

class MobileNetV3Stem(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()
        
        self.conv = nn.LazyConv2d(kernel_size=3, out_channels=out_channels,
                                  bias=False,
                                  stride=2, padding=1,
                                  )
        self.hswish = nn.Hardswish()
        self.bn = nn.LazyBatchNorm2d()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hswish(x)
        return x
    
            
class MobileNetV3InventedResidualBlock(nn.Module):
    def __init__(self, out_channels, width_multiplier, expansion_size,
                 non_linearity: Literal["relu", "hswish"],
                 stride, depthwiseconv_kernel_size: Literal["3x3", "5x5"],
                 use_squeeze_excitation: bool,
                 batch_norm: bool,
                 **kwargs
                ):
        super().__init__()
        out_channels = int(out_channels * width_multiplier) if width_multiplier else out_channels
        
        self.expansion_conv = nn.LazyConv2d(out_channels=expansion_size, kernel_size=1,
                                            #padding=1, 
                                            bias=False
                                            )
        self.bn1 = nn.LazyBatchNorm2d() if batch_norm else nn.Identity()            
        
        if depthwiseconv_kernel_size == "3x3":
            self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None, 
                                                      kernel_size=3,
                                                      bias=False,
                                                      stride=stride, #1,
                                                      padding=1
                                                    )
            if use_squeeze_excitation:
                squeeze_exite = SqueezeExcitation(reduction_ratio=kwargs.get("reduction_ratio", 4))
                self.depthwise_conv = nn.Sequential(self.depthwise_conv, squeeze_exite)
                            
        elif depthwiseconv_kernel_size == "5x5":
            self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None, kernel_size=5,
                                                      stride=1, bias=False,
                                                      padding=2
                                                      )
            if use_squeeze_excitation:
                squeeze_exite = SqueezeExcitation(reduction_ratio=kwargs.get("reduction_ratio", 4))
                self.depthwise_conv = nn.Sequential(self.depthwise_conv, squeeze_exite)
                
        self.pointwise_conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=1,
                                            #padding=1, 
                                            bias=False
                                            )
        self.bn2 = nn.LazyBatchNorm2d() if batch_norm else nn.Identity()
        self.bn3 = nn.LazyBatchNorm2d() if batch_norm else nn.Identity()
        
        if non_linearity == "relu":
            self.act1 = nn.ReLU6()
            self.act2 = nn.ReLU6()
            self.act3 = nn.ReLU6()
        elif non_linearity == "hswish":
            self.act1 = nn.Hardswish()
            self.act2 = nn.Hardswish()
            self.act3 = nn.Hardswish()
            
    def forward(self, x):
        shortcut = x
        x = self.expansion_conv(x)
        x = self.bn1(x)
        self.act1(x)
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        #x = self.act3(x)
        x += shortcut
        return x
    
    
class MobileNetV3InventedNonResidualBlock(nn.Module):
    def __init__(self, *, out_channels, width_multipler, expansion_size,
                 non_linearity: Literal["relu", "hswish"],
                depthwiseconv_kernel_size: Literal["3x3", "5x5"],
                use_squeeze_excitation: bool,
                batch_norm: bool,
                stride=2,
                **kwargs
                ):
        super().__init__()
        if non_linearity not in NON_LINEARITY_VALUES:
            raise ValueError(f"non_linearity: {non_linearity} was provided but expected to be one of {NON_LINEARITY_VALUES}")
        
        if depthwiseconv_kernel_size not in KERNEL_SIZE_VALUES:
            raise ValueError(f"depthwiseconv_kernel_size: {depthwiseconv_kernel_size} is not a valid value. It must be one of {KERNEL_SIZE_VALUES}")
        
        out_channels = int(out_channels * width_multipler) if width_multipler else out_channels
        
        self.expansion_conv = nn.LazyConv2d(out_channels=expansion_size,
                                            kernel_size=1,# padding=1,
                                            bias=False
                                            )
        if depthwiseconv_kernel_size == "3x3":
            self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None, 
                                                      kernel_size=3,
                                                      stride=stride,
                                                      bias=False, padding=1
                                                      )
            if use_squeeze_excitation:
                squeeze_exite = SqueezeExcitation(reduction_ratio=kwargs.get("reduction_ratio", 4))
                self.depthwise_conv = nn.Sequential(self.depthwise_conv, squeeze_exite)
                
        elif depthwiseconv_kernel_size == "5x5":
            self.depthwise_conv = LazyDepthwiseConv2d(out_channels=None,
                                                      kernel_size=5,
                                                      stride=stride,
                                                      padding=2, 
                                                      bias=False
                                                      )
            
            if use_squeeze_excitation:
                squeeze_exite = SqueezeExcitation(reduction_ratio=kwargs.get("reduction_ratio", 4))
                self.depthwise_conv = nn.Sequential(self.depthwise_conv, squeeze_exite)
                            
        self.pointwise_conv = nn.LazyConv2d(out_channels=out_channels,
                                            kernel_size=1,
                                            #padding=1, 
                                            bias=False
                                            )
        
        if non_linearity == "relu":
            self.act1 = nn.ReLU6()
            self.act2 = nn.ReLU6()
            self.act3 = nn.ReLU6()
            
        elif non_linearity == "hswish":
            self.act1 = nn.Hardswish()
            self.act2 = nn.Hardswish()
            self.act3 = nn.Hardswish()
            
        self.bn1 = nn.LazyBatchNorm2d() if batch_norm else nn.Identity()
        self.bn2 = nn.LazyBatchNorm2d() if batch_norm else nn.Identity()
        self.bn3 = nn.LazyBatchNorm2d() if batch_norm else nn.Identity()
            
    def forward(self, x):
        x = self.expansion_conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        #x = self.act3(x)
        return x
 
 
class SELazyLinear(nn.LazyLinear):
    def __init__(self, reduction_ratio,
                 mode: Literal["reduce", "expand"],
                 bias=True
                 ):
        super().__init__(out_features=None)
        self.reduction_ratio = reduction_ratio
        self.mode = mode
        
    def initialize_parameters(self, input):
        x = input[0] if isinstance(input, (list, tuple)) else input
        
        in_ch = int(x.shape[1])
        self.in_features = in_ch
                
        if self.mode == "reduce":
            self.out_features = int(max(1, in_ch // self.reduction_ratio))
        elif self.mode == "expand":
            self.out_features = int(self.in_features * self.reduction_ratio)
        super().initialize_parameters(input)
        
               
class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
    def forward(self, x):
        x = self.global_avgpool(x)
        return x
        
                    
class Excitation(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.fc1 = SELazyLinear(reduction_ratio=reduction_ratio, 
                                mode="reduce",
                               bias=False
                               )
        self.relu = nn.ReLU()   
        self.fc2 = SELazyLinear(reduction_ratio=reduction_ratio,
                                mode="expand", 
                                bias=False
                                )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
                     

class SqueezeExcitation(nn.Module):
    def __init__(self, reduction_ratio):
        super().__init__()
        self.squeeze = Squeeze()
        self.excitation = Excitation(reduction_ratio=reduction_ratio)
        self.calibrate = nn.Sigmoid()
        
    def forward(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = torch.flatten(x, start_dim=1)
        x = self.excitation(x)
        x = self.calibrate(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        recalibrated_x = shortcut * x
        return recalibrated_x


class MobileNetV3ConvBlock(nn.Module):
    def __init__(self, out_channels, pool: bool, batch_norm: bool, non_linearity="hswish", **kwargs):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)) if pool else nn.Identity()
        self.bn = nn.LazyBatchNorm2d() if batch_norm else nn.Identity()
        self.conv = nn.LazyConv2d(out_channels=out_channels,
                                  kernel_size=1,
                                  bias=False
                                  )
        
        if non_linearity == "relu":
            self.act1 = nn.ReLU6()
        elif non_linearity == "hswish":
            self.act1 = nn.Hardswish()
            
    def forward(self, x):
        x = self.global_avgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act1(x)
        return x
        
        
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=num_classes,
                                  kernel_size=1,
                                  bias=False
                                  )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x
    

class BlockConfig(NamedTuple):
    num_blocks: int
    out_channels: int
    non_linearity: Literal[NON_LINEARITY_VALUES]    
    use_squeeze_excitation: bool
    expansion_sizes: List
    strides: int
    depthwiseconv_kernel_size: Literal["3x3", "5x5"]
    invented_residual: bool = True
    batch_norm: bool = True
    pool: bool =False
        
    
class MobileNetV3GroupConfig(NamedTuple):
    width_multiplier: int
    block_config: List[BlockConfig]
    
    
def group(*, out_channels, width_multiplier,
          expansion_sizes, non_linearity,
          use_squeeze_excitation,
          num_blocks, strides,
          **kwargs,
          ):
    blocks = []
        
    if num_blocks != len(expansion_sizes):
        raise ValueError(f"num_blocks: {num_blocks} != expansion_sizes: {len(expansion_sizes)}. Number of num_blocks must equal number of items in expansion_sizes")
    
    for idx, (expansion_size, stride, out_channel) in enumerate(zip(expansion_sizes, strides, out_channels)):
        if len(out_channels) == 1:
            if stride == 1:
                bottleneck = MobileNetV3InventedResidualBlock(out_channels=out_channel, width_multiplier=width_multiplier,
                                                                expansion_size=expansion_size,
                                                                use_squeeze_excitation=use_squeeze_excitation,
                                                                stride=stride,
                                                                non_linearity=non_linearity, 
                                                                **kwargs
                                                                )
            else:
                bottleneck = MobileNetV3InventedNonResidualBlock(out_channels=out_channel,
                                                                 width_multipler=width_multiplier,
                                                                 expansion_size=expansion_size,
                                                                 use_squeeze_excitation=use_squeeze_excitation,
                                                                 stride=stride,
                                                                 non_linearity=non_linearity,
                                                                 **kwargs
                                                                 )
        else:
            prev_out_channel = out_channels[idx-1]
            
            if (idx == 0) or not (out_channel == prev_out_channel and stride == 1):
                bottleneck = MobileNetV3InventedNonResidualBlock(out_channels=out_channel,
                                                                 width_multipler=width_multiplier,
                                                                 expansion_size=expansion_size,
                                                                 non_linearity=non_linearity,
                                                                 use_squeeze_excitation=use_squeeze_excitation,
                                                                 stride=stride,
                                                                 **kwargs
                                                                 )
            
            elif  out_channel == prev_out_channel and stride == 1:
                bottleneck = MobileNetV3InventedResidualBlock(out_channels=out_channel,
                                                              width_multiplier=width_multiplier,
                                                              expansion_size=expansion_size,
                                                              non_linearity=non_linearity,
                                                              stride=stride,
                                                              use_squeeze_excitation=use_squeeze_excitation,
                                                              **kwargs
                                                              )
            
        blocks.append(bottleneck)
    return nn.Sequential(*blocks)

def learner(configs):
    grp_blks = []
    for idx, conf in enumerate(configs.block_config):
        if conf.invented_residual == False:
            for i in range(conf.num_blocks):
                conv = MobileNetV3ConvBlock(**conf._asdict())   
                grp_blks.append(conv) 
        else:
            grp_blks.append(group(**conf._asdict(), **configs._asdict(), #prev_out_channel=prev_out_channel
                                  ))
    return nn.Sequential(*grp_blks)            


def make_model(num_classes, learner_configs, data, device="cuda", 
               initializer_type="he_normal"
               ):
    stem = MobileNetV3Stem()
    learner_module = learner(configs=learner_configs)
    classifier = Classifier(num_classes=num_classes)
    model = nn.Sequential(stem, learner_module, classifier).to(device)
    data.to(device)
    _ = model(data)
    model.apply(lambda module: kernel_initializer(module, initializer_type=initializer_type))
    return model
    

large_block_config = [BlockConfig(out_channels=[16], num_blocks=1, non_linearity="relu", 
                            use_squeeze_excitation=False, expansion_sizes=[16], 
                            depthwiseconv_kernel_size="3x3",
                            strides=[1]),
                BlockConfig(out_channels=[24, 24], depthwiseconv_kernel_size="3x3", 
                            num_blocks=2, non_linearity="relu", 
                            use_squeeze_excitation=False, expansion_sizes=[64, 72],
                            strides=[2, 1]
                            ),
                BlockConfig(out_channels=[40, 40, 40], depthwiseconv_kernel_size="5x5", num_blocks=3, 
                            non_linearity="relu", use_squeeze_excitation=True, 
                            expansion_sizes=[72, 120, 120],
                            strides=[2,1,1]
                            ),
                BlockConfig(out_channels=[80, 80, 80, 80], depthwiseconv_kernel_size="3x3", num_blocks=4, 
                            non_linearity="hswish", use_squeeze_excitation=False, 
                            expansion_sizes=[240, 200, 184, 184],
                            strides=[2, 1, 1, 1]
                            ),
                BlockConfig(out_channels=[112, 112], depthwiseconv_kernel_size="3x3", num_blocks=2, 
                            non_linearity="hswish", use_squeeze_excitation=True, 
                            expansion_sizes=[480, 672],
                            strides=[1, 1]
                            ),
                BlockConfig(out_channels=[160, 160, 160], depthwiseconv_kernel_size="5x5", num_blocks=3, 
                            non_linearity="hswish", use_squeeze_excitation=True, 
                            expansion_sizes=[672, 960, 960],
                            strides=[2, 1, 1]
                            ),
                BlockConfig(invented_residual=False, 
                            depthwiseconv_kernel_size=None, out_channels=960, 
                            num_blocks=1, non_linearity="hswish", use_squeeze_excitation=False, 
                            expansion_sizes=[None],
                            strides=[1]),
                BlockConfig(invented_residual=False, 
                            depthwiseconv_kernel_size=None, out_channels=1280, 
                            num_blocks=1, non_linearity="hswish", use_squeeze_excitation=False,
                            batch_norm=False, expansion_sizes=[None],
                            strides=[1], pool=True,
                            )
                ]
                
 
large_group_config = MobileNetV3GroupConfig(width_multiplier=1, block_config=large_block_config) 

if __name__ == "__main__":
    device="cuda"
    data = torch.randn(1, 3, 224, 224).to(device)
    model = make_model(num_classes=1000, learner_configs=large_group_config, 
                       data=data,
                       device=device,
                       initializer_type="he_normal"
                       )
    
    summary(model=model, input_data=data, depth=4,
            col_names=["input_size", "output_size", "num_params", "mult_adds"]
            )

