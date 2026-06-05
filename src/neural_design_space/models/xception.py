import torch
import torch.nn as nn
from typing import NamedTuple
from torchinfo import summary
from neural_design_space.utils import kernel_initializer


class XceptionStem(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.conv1 = nn.Sequential(
                                nn.LazyConv2d(32, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(inplace=True)
                            )
        
        self.conv2 = nn.Sequential(nn.LazyConv2d(out_channels=64,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                   nn.LazyBatchNorm2d(),
                                   nn.ReLU(inplace=True)
                                   )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class LazyDepthwiseConv2d(nn.LazyConv2d):
    def initialize_parameters(self, input):
        # Let LazyConv2d infer in_channels and out_channels
        super().initialize_parameters(input)

        # Now that in_channels is known, set groups = in_channels
        self.groups = self.in_channels

        self.weight = nn.Parameter(torch.empty(self.in_channels, 1, 
                                                *self.kernel_size, 
                                                device=input.device,
                                                dtype=input.dtype
                                                )
                                    )
        
        # Depthwise conv requires out_channels == in_channels
        if self.out_channels != self.in_channels:
            raise ValueError(
                f"Depthwise conv requires out_channels == in_channels, "
                f"got out={self.out_channels}, in={self.in_channels}"
            )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.depthwise_conv = nn.Sequential(
                                            LazyDepthwiseConv2d(out_channels=in_channels, 
                                                                kernel_size=3,
                                                                stride=1, padding=1,
                                                                bias=False
                                                                ),
                                            nn.LazyBatchNorm2d()
                                            )
        
        self.pointwise_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels, 
                                                        kernel_size=1,
                                                        stride=1, padding=0,
                                                        bias=False
                                                        ),
                                            nn.LazyBatchNorm2d()
                                            )
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    
    
class ProjectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 skip_first_relu=False
                 ):
        super().__init__()
        
        self.proj_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels,
                                                     kernel_size=1,stride=2,
                                                     bias=False, padding=0
                                                     ),
                                            nn.LazyBatchNorm2d()
                                            )
        self.relu = nn.ReLU(inplace=True)
        self.separable_conv1 = DepthwiseSeparableConv(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      )
        self.separable_conv2 = DepthwiseSeparableConv(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        block = []
        if not skip_first_relu:
            block.append(nn.ReLU(inplace=True))
        block.extend([self.separable_conv1, 
                      self.relu, 
                      self.separable_conv2, 
                      self.maxpool
                      ]
                     )
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        shortcut = self.proj_conv(x)
        x = self.block(x)
        x += shortcut
        return x    
        
class EntryFlow(nn.Module):
    def __init__(self, out_channels=[128, 256, 728]):
        super().__init__()
        
        self.stem = XceptionStem()
        blocks = []
        for idx, chn in enumerate(out_channels):
            block = ProjectionBlock(in_channels=64 if idx==0 else out_channels[idx-1],
                                    out_channels=chn,
                                    skip_first_relu=(idx==0)
                                    )
            blocks.append(block)
        self.separable_blocks_with_skip = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.separable_blocks_with_skip(x)
        return x

    
class MiddleFlowIdentityModule(nn.Module):
    def __init__(self, in_channels, out_channels,
                 module_depth=3
                 ):
        super().__init__()
        
        middleflow_module = []
                
        middleflow_module.append(nn.ReLU(inplace=True))
        middleflow_module.append(DepthwiseSeparableConv(in_channels=in_channels, 
                                                        out_channels=out_channels,
                                                        )
                                 )
        
        
        for i in range(module_depth-1):
            middleflow_module.append(nn.ReLU(inplace=True))
            middleflow_module.append(DepthwiseSeparableConv(in_channels=out_channels, 
                                                            out_channels=out_channels,
                                                            )
                                    )
            
        self.middleflow_module = nn.Sequential(*middleflow_module)
        
    def forward(self, x):
        shortcut = x
        x = self.middleflow_module(x)
        x += shortcut
        return x       
        
class MiddleFlowBlock(nn.Module):
    def __init__(self, in_channels=728, out_channels=728,
                 module_depth=3, 
                 n_blocks=8
                 ):
        super().__init__()
        
        blocks = [MiddleFlowIdentityModule(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         module_depth=module_depth,
                                         )
                  for _ in range(n_blocks)
                  ]
        
        self.middleflow_blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.middleflow_blocks(x)
        return x
        
        
class ExitFlow(nn.Module):
    def __init__(self, in_channels=728, out_channels=1024):
        super().__init__()
        self.proj_conv = nn.Sequential(nn.LazyConv2d(out_channels=out_channels,
                                                     kernel_size=1,
                                                     stride=2, bias=False,
                                                     padding=0,
                                                     ),
                                        nn.LazyBatchNorm2d()
                                        )
        self.separable_conv1 = DepthwiseSeparableConv(in_channels=in_channels, 
                                                      out_channels=in_channels,
                                                      )
        self.separable_conv2 = DepthwiseSeparableConv(in_channels=in_channels, 
                                                      out_channels=out_channels,
                                                      )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.exitflow_module1 = nn.Sequential(nn.ReLU(inplace=True),
                                            self.separable_conv1,
                                            nn.ReLU(inplace=True),
                                            self.separable_conv2,
                                            self.maxpool
                                            )
        
        self.separable_conv3 = DepthwiseSeparableConv(in_channels=out_channels, 
                                                      out_channels=1536,
                                                      )
        self.separable_conv4 = DepthwiseSeparableConv(in_channels=1536,
                                                      out_channels=2048,
                                                      )
        
        self.exitflow_module2 = nn.Sequential(self.separable_conv3, 
                                              nn.ReLU(inplace=True), 
                                              self.separable_conv4,
                                              nn.ReLU(inplace=True),
                                              nn.AdaptiveAvgPool2d((1,1))
                                              )
        
    def forward(self, x):
        shortcut = self.proj_conv(x)
        x = self. exitflow_module1(x)
        x += shortcut
        
        x = self.exitflow_module2(x)
        return x
    
    
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.LazyLinear(out_features=num_classes),
                                        nn.Softmax(dim=1)
                                        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x
                
                
def make_model(data, config, device="cuda"):
    entryflow = EntryFlow(out_channels=config.entryflow_out_channels)
    middleflow = MiddleFlowBlock(in_channels=config.entryflow_out_channels[-1],
                                 out_channels=config.entryflow_out_channels[-1],
                                 module_depth=config.middleflow_module_depth,
                                 n_blocks=config.middleflow_n_blocks,
                                 )
    exitflow = ExitFlow(in_channels=config.entryflow_out_channels[-1])
    classifier = Classifier(num_classes=config.num_classes)
    model = nn.Sequential(entryflow, middleflow, exitflow, classifier)   
    model.to(device)
    _ = model(data.to(device))
    model.apply(kernel_initializer)
    return model

                
class XceptionConfig(NamedTuple):
    num_classes: int
    entryflow_out_channels: list = [128, 256, 728]
    middleflow_module_depth: int = 3
    middleflow_n_blocks: int = 8


if __name__ == "__main__":
    
    config = XceptionConfig(num_classes=1000,
                            entryflow_out_channels=[128, 256, 728],
                            middleflow_module_depth=3,
                            middleflow_n_blocks=8
                            )    
    data = torch.randn(1, 3, 299, 299)

    model = make_model(data=data, config=config, device="cuda")
    
    print(f"Custom Xception model summary:\n")
    summary(model, input_data=data, device='cuda',
            verbose=1,
            mode="train",
            col_names=["input_size", "output_size", "num_params",
                       "mult_adds",
                       ],
            depth=3,
            )


        