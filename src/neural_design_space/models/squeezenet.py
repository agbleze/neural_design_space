import torch
import torch.nn as nn
from torchinfo import summary
from typing import NamedTuple, List, Tuple, Optional, Literal
from neural_design_space.utils.utils import kernel_initializer


class SqueezeNetStem(nn.Module):
    def __init__(self, out_channels=96, downsample: bool=True, **kwargs):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=7, stride=2,
                                  bias=False
                                  )
        self.downsample_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()
                
    def forward(self, x):
        x = self.conv(x)
        x = self.downsample_layer(x)
        return x


class FireModule(nn.Module):
    def __init__(self, s1x1, e1x1, e3x3, dropout=None, **kwargs):
        super().__init__()
        self.squeeze = nn.LazyConv2d(out_channels=s1x1, kernel_size=1, bias=False,
                                     )
        self.expand1x1 = nn.LazyConv2d(out_channels=e1x1, kernel_size=1, bias=False,
                                       )
        self.expand3x3 = nn.LazyConv2d(out_channels=e3x3, kernel_size=3, padding=1, bias=False,
                                       )
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        
    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1)
        x = self.dropout(x)
        return x

class FireIdentityModule(nn.Module):
    def __init__(self, s1x1, e1x1, e3x3, dropout=None, **kwargs):
        super().__init__()
        self.fire_module = FireModule(s1x1=s1x1, e1x1=e1x1, e3x3=e3x3, dropout=dropout, **kwargs)
        
    def forward(self, x):
        identity = x
        out = self.fire_module(x)
        out += identity
        return out   

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv = nn.LazyConv2d(out_channels=num_classes, kernel_size=1, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x
    
    
class SqueezeNetBlockConfig(NamedTuple):
    s1x1: int
    e1x1: int
    e3x3: int
    num_blocks: int
    downsample: bool = False
    
    
class SqueezeNetGroupConfig(NamedTuple):
    block_configs: List[SqueezeNetBlockConfig]
    bypass_type: Optional[Literal["simple", "complex"]] = None
  
    
def group(*, s1x1, e1x1, e3x3, num_blocks, downsample: bool=False,
          bypass_type: Optional[Literal["simple", "complex"]]=None,
          **kwargs
          ):
    
    blocks = []
    for i in range(num_blocks):
        if i == 1:
            mod = FireIdentityModule(s1x1=s1x1, e1x1=e1x1, e3x3=e3x3) if bypass_type == "simple" else FireModule(s1x1=s1x1, e1x1=e1x1, e3x3=e3x3)
            if downsample:
                #mod = FireModule(s1x1=s1x1, e1x1=e1x1, e3x3=e3x3)
                mod = nn.Sequential(mod, nn.MaxPool2d(kernel_size=3, stride=2))
        else:
            mod = FireModule(s1x1=s1x1, e1x1=e1x1, e3x3=e3x3)
        blocks.append(mod)
    return nn.Sequential(*blocks)
    
    
def learner(group_config: SqueezeNetGroupConfig):
    grpoup_modules = []
    for block_config in group_config.block_configs:
        grp = group(**block_config._asdict())
        grpoup_modules.extend(grp)
    return nn.Sequential(*grpoup_modules)



def make_model(num_classes, learner_configs: SqueezeNetGroupConfig, 
               data, device="cpu", initializer_type="he_normal"
               ):
    stem = SqueezeNetStem()
    learner_module = learner(learner_configs)
    classifier = Classifier(num_classes=num_classes)
  
    data = data.to(device)
    model = nn.Sequential(stem, learner_module, classifier).to(device)
    _ = model(data)  
    
    model.apply(lambda m: kernel_initializer(m, initializer_type=initializer_type))
    model = model.to(device)
    return model


if __name__ == "__main__":
    device="cuda"
    data = torch.randn(1, 3, 224, 224).to(device)
    
    block_config = [SqueezeNetBlockConfig(s1x1=16, e1x1=64, e3x3=64, num_blocks=2, downsample=False),
                    SqueezeNetBlockConfig(s1x1=32, e1x1=128, e3x3=128, num_blocks=2, downsample=True),
                    SqueezeNetBlockConfig(s1x1=48, e1x1=192, e3x3=192, num_blocks=2, downsample=False),
                    SqueezeNetBlockConfig(s1x1=64, e1x1=256, e3x3=256, num_blocks=2, downsample=True)
                    ]
    group_config = SqueezeNetGroupConfig(block_configs=block_config, bypass_type="simple")
    
    model = make_model(num_classes=1000, learner_configs=group_config, 
                        data=data,
                        device=device,
                        initializer_type="he_normal"
                        )
    summary(model, input_size=(1, 3, 224, 224))