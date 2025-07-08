import torch
import torch.nn as nn
import torch.nn.functional as F  

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        
       
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu1 = nn.GELU()
        
       
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.gelu2 = nn.GELU()
    
    def forward(self, x):
       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        
        return x

class IRB(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(IRB, self).__init__()
        
        self.use_residual = in_channels == out_channels
        expanded_channels = in_channels * expansion_factor
        
      
        self.expand = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(expanded_channels),
            nn.GELU()
        )
        
     
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels=expanded_channels,
                out_channels=expanded_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=expanded_channels  
            ),
            nn.BatchNorm1d(expanded_channels),
            nn.GELU()
        )
        
    
        self.project = nn.Sequential(
            nn.Conv1d(
                in_channels=expanded_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
   
        x = self.expand(x)
  
        x = self.depthwise(x)
        

        x = self.project(x)
        
   
        if self.use_residual:
            x = x + identity
            
        return x

class MSDPM(nn.Module):
    def __init__(self, channels):
        super(MSDPM, self).__init__()
        
        
        self.initial_dw = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                padding=3,
                groups=channels
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        

        self.dilated_conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=2,
                dilation=2
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        

        self.dilated_conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=3,
                dilation=3
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        
  
        self.final_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1
            ),
            nn.BatchNorm1d(channels)
        )
        
        self.gelu = nn.GELU()
        
    def forward(self, x):
 
        y = self.initial_dw(x)
        
 
        z2 = self.dilated_conv1(y)
        z3 = self.dilated_conv2(y)
        
  
        z = self.gelu(z2 + z3)
        
   
        out = self.final_conv(z)
        return out

class LK_FFN(nn.Module):
    def __init__(self, channels, expansion_factor=4):
        super(LK_FFN, self).__init__()
        
        hidden_dim = channels * expansion_factor
        
      
        self.dw_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                padding=3,
                groups=channels
            ),
            nn.BatchNorm1d(channels)
        )
        

        self.fc1 = nn.Conv1d(channels, hidden_dim, 1)  
        self.fc2 = nn.Conv1d(hidden_dim, channels, 1)  
        self.bn = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        
    def forward(self, z):

        x = self.dw_conv(z)
        

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.bn(x)
        
        return x

class DCB(nn.Module):
    def __init__(self, channels, expansion_factor=4):
        super(DCB, self).__init__()
        
        self.msdpm = MSDPM(channels)
        self.lk_ffn = LK_FFN(channels, expansion_factor)
        
    def forward(self, x):

        x = x + self.msdpm(x)  
        

        x = x + self.lk_ffn(x)  
        
        return x

class TinyLN(nn.Module):
    def __init__(self,
                 in_channels=1,     
                 stem_channels=64,   
                 num_classes=2,      
                 expansion_factor=4,  
                 num_irb_blocks=1,     
                 num_dcb_blocks=1    
                ):
        super(TinyLN, self).__init__()
        
 
        self.stem = Stem(in_channels, stem_channels)
        
        # Inverted Residual Blocks
        self.irb_blocks = nn.Sequential()
        for i in range(num_irb_blocks):
            self.irb_blocks.add_module(
                f"irb_block_{i + 1}",
                IRB(
                    in_channels=stem_channels,
                    out_channels=stem_channels,
                    expansion_factor=expansion_factor
                )
            )
        
 
        self.dcb_blocks = nn.Sequential()
        for i in range(num_dcb_blocks):
            self.dcb_blocks.add_module(
                f"dcb_block_{i + 1}",
                DCB(
                    channels=stem_channels,
                    expansion_factor=expansion_factor
                )
            )
        
 
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
   
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stem_channels, num_classes)
        )
        
    def forward(self, x, return_features=False):
  
        x = self.stem(x)
        
        # Inverted Residual Blocks
        for irb_block in self.irb_blocks:
            x = irb_block(x)
        
 
        for dcb_block in self.dcb_blocks:
            x = dcb_block(x)
        

        x = self.global_pool(x)
        
 
        features = x.view(x.size(0), -1)
        
 
        output = self.classifier(features)
        
        if return_features:
            return output, features
        return output

if __name__ == "__main__":

    model = TinyLN(
        in_channels=1,
        stem_channels=64,
        num_classes=2,
        expansion_factor=4,
        num_irb_blocks=1,
        num_dcb_blocks=1
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)


    x = torch.randn(32, 1, 1024).to(device) 

    output, features = model(x, return_features=True)
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")

