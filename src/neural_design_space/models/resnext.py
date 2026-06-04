import torch.nn as nn
import torch
from torchinfo import summary
import torchvision.models as models
from neural_design_space.utils.utils import kernel_initializer

class ResNextStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, 
                                  bias=False, padding=3
                                  )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)
        return x
        

class ResNextClassifier(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(out_features=num_classes)   
        self.softmax = nn.Softmax(dim=1)     
        
    def forward(self, x):
        x = self.global_avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
        
        
class ResNextIdentityBlock(nn.Module):
    def __init__(self, filter_in, filter_out, cardinality=32):
        super().__init__() 
        
        # 1x1 Dimensionality reduction
        self.reduce = nn.Sequential(
                                    nn.LazyConv2d(out_channels=filter_in, kernel_size=1, stride=1, padding="same", bias=False),
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU()
                                )
        
        # Cardinality (wide) Layer split-transform
        self.group = nn.Sequential(nn.LazyConv2d(out_channels=filter_in, kernel_size=3, stride=1, padding="same", bias=False, 
                                                 groups=cardinality
                                                ),
                                   nn.LazyBatchNorm2d(),
                                   )
        self.relu = nn.ReLU()
        
        # 1x1 expansion 
        self.expand = nn.Sequential(nn.LazyConv2d(out_channels=filter_out,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding="same", bias=False
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU()
                                    )
        
        
    def forward(self, x):
        shortcut = x
        x = self.reduce(x)
        x = self.group(x)
        x = self.expand(x)
        x += shortcut
        x = self.relu(x)
        return x
        

class ResNextProjectionBlock(nn.Module):
    def __init__(self, filters_in, filters_out, cardinality=32, strides=2, **kwargs):
        super().__init__(**kwargs)
        # construct projection shortcut layer
        self.proj = nn.Sequential(nn.LazyConv2d(out_channels=filters_out, kernel_size=1,
                                                stride=strides,
                                                ),
                                  nn.LazyBatchNorm2d(),
                                  )
        
        # Dimensionality reduction
        self.reduce = nn.Sequential(nn.LazyConv2d(out_channels=filters_in, 
                                                  kernel_size=1,
                                                  stride=1, padding="same", bias=False
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    nn.ReLU()
                                    )
        
        # Cardinality (wide) Layer split-transform
        self.group_conv = nn.Sequential(nn.LazyConv2d(out_channels=filters_in,
                                                      kernel_size=3,
                                                      stride=strides,
                                                      padding=1,
                                                      bias=False,
                                                      groups=cardinality
                                                      ),
                                        nn.LazyBatchNorm2d(),
                                        nn.ReLU()
                                        )
        # 1x1 expansion dimensionality restoration
        self.expand = nn.Sequential(nn.LazyConv2d(out_channels=filters_out,
                                                  kernel_size=1, stride=1,
                                                  padding="same", bias=False
                                                  ),
                                    nn.LazyBatchNorm2d(),
                                    )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        shortcut = self.proj(x)
        x = self.reduce(x)
        x = self.group_conv(x)
        x = self.expand(x)
        x += shortcut
        x = self.relu(x)
        return x
        
    
def group(filters_in, filters_out, n_blocks, cardinality=32, strides=2):
    block_collection = []
    block = ResNextProjectionBlock(filters_in=filters_in, filters_out=filters_out,
                               cardinality=cardinality, strides=strides
                               )
    block_collection.append(block)
    for _ in range(n_blocks -1):
        block = ResNextIdentityBlock(filter_in=filters_in, filter_out=filters_out, 
                                     cardinality=cardinality)
        block_collection.append(block)
    
    return nn.Sequential(*block_collection)
    
    
    
def learner(groups, cardinality=32):
    group_col = []
    filters_in, filters_out, n_blocks = groups.pop(0)
    x = group(filters_in=filters_in, filters_out=filters_out, n_blocks=n_blocks, strides=1, cardinality=cardinality)
    group_col.append(x)
    for filters_in, filters_out, n_blocks in groups:
        x = group(filters_in=filters_in, filters_out=filters_out, n_blocks=n_blocks, cardinality=cardinality)
        group_col.append(x)
    return nn.Sequential(*group_col)

if __name__ == "__main__":

    groups = {50: [(128, 256, 3), (256, 512, 4), (512, 1024, 6), (1024, 2048, 3)],
            101: [(128, 256, 3), (256, 512, 4), (512, 1024, 23), (1024, 2048, 3)],
            152: [(128, 256, 3), (256, 512, 8), (512, 1024, 36), (1024, 2048, 3)]
            }

    cardinality = 32

    stem = ResNextStem()
    learner_module = learner(groups=groups[50], cardinality=cardinality)
    classifier = ResNextClassifier(num_classes=1000)

    model = nn.Sequential(stem, learner_module, classifier).to("cuda")
    example_input = torch.randn(1, 3, 224, 224, device="cuda")

    _ = model(example_input)
    model.apply(kernel_initializer)

    #from torchsummary import summary
    print(f"Custom ResNext50 model summary:\n{summary(model, input_data=example_input)}")

    print("Default ResNext50 model summary:")
    resnext50 = models.resnext50_32x4d(weights=None).to("cuda")
    print(f"Default ResNext50 model summary:\n{summary(resnext50, input_data=example_input)}")
    
    models.vit_b_16(weights=None).to("cuda")