import torch
import torch.nn as nn
from torchinfo import summary
from neural_design_space.utils.utils import kernel_initializer




class Encoder(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        
        encoder_layers = []
        for out_ch in layers:
            encoder_layer = nn.Sequential(nn.LazyConv2d(out_channels=out_ch, 
                                                        kernel_size=3, stride=2, 
                                                        padding=1, 
                                                        bias=False
                                                        ),
                                          nn.LazyBatchNorm2d(),
                                          nn.ReLU()
                                          )
            encoder_layers.append(encoder_layer)
            self.add_module(f"encoder_layer_{out_ch}", encoder_layer)
        self.encoder = nn.Sequential(*encoder_layers)
        
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        decoder_layers = []
        decoder_layers_out_ch = layers[::-1]
        
        for out_ch in decoder_layers_out_ch:
            decoder_layer = nn.Sequential(nn.LazyConvTranspose2d(out_channels=out_ch,
                                                                 kernel_size=3, stride=2, padding=1, 
                                                                 output_padding=1, 
                                                                 bias=False
                                                                 ),
                                        nn.LazyBatchNorm2d(), 
                                        nn.ReLU()
                                        )
            decoder_layers.append(decoder_layer)
        self.decoder = nn.Sequential(*decoder_layers)
        #self.decoder_activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.decoder(x)
        #x = self.decoder_activation(x)
        return x
    

class ReconstructTask(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.task_layer = nn.Sequential(nn.LazyConvTranspose2d(out_channels=out_channels,
                                                               kernel_size=3, stride=2, padding=1, 
                                                              output_padding=1, bias=False
                                                              ),
                                        nn.Sigmoid()
                                        )
        
    def forward(self, x):
        x = self.task_layer(x)
        return x
    
    
    
class AutoEncoder(nn.Module):
    def __init__(self, layer_channels: list[int], out_channels=3):
        super().__init__()
        self.encoder = Encoder(layer_channels)
        self.decoder = Decoder(layer_channels)
        self.reconstruct_task = ReconstructTask(out_channels)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.reconstruct_task(x)
        return x




def make_model(layer_channels, data, device="cuda", 
               initializer_type="he_normal"
               ):
    out_channels = data.shape[1]
    model = AutoEncoder(layer_channels=layer_channels, out_channels=out_channels).to(device)
    data.to(device)
    _ = model(data)
    model.apply(lambda module: kernel_initializer(module, initializer_type=initializer_type))
    return model


data = torch.randn(1, 3, 224, 224)
layer_channels = [64, 128, 256, 512]
model = make_model(layer_channels=layer_channels, data=data, device="cuda", initializer_type="he_normal")
summary(model, input_size=data.shape)



