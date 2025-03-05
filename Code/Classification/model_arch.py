import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet50

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape to sequence: (B, C, H, W) -> (HW, B, C)
        x_seq = x.flatten(2).permute(2, 0, 1)
        
        # Self-attention block
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        # MLP block
        x_norm = self.norm2(x_seq)
        mlp_out = self.mlp(x_norm)
        x_seq = x_seq + mlp_out
        
        # Reshape back: (HW, B, C) -> (B, C, H, W)
        x = x_seq.permute(1, 2, 0).reshape(B, C, H, W)
        
        return x

class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(pool))
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        self.channels = channels
        self.last_attention = None
        
    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = (avg_out + max_out).view(b, c, 1, 1)
        self.last_attention = attention
        return x * attention

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * scale

class StochasticDepth(nn.Module):
    def __init__(self, prob, mode='batch'):
        super().__init__()
        self.prob = prob
        self.mode = mode
        self.training = True

    def forward(self, x):
        if not self.training or self.prob == 0:
            return x
        
        if self.mode == 'row':
            shape = [x.shape[0]] + [1] * (x.dim() - 1)
        else:
            shape = [1] * x.dim()

        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) > self.prob
        random_tensor = random_tensor.to(x.dtype) / (1 - self.prob)
        
        return x * random_tensor

class HybridModel(nn.Module):
    def __init__(self, num_classes=5, stochastic_depth_prob=0.1, use_residual=False, 
                 use_skip_connections=False):
        super().__init__()
        
        # Modified ResNet50 backbone with reduced initial layers
        resnet = resnet50(weights='IMAGENET1K_V1')
        
        # Modify first conv layer for better feature extraction
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Only freeze batch norm layers, allow conv layers to adapt
        for module in resnet.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        
        # Enhanced encoder with balanced attention
        self.encoder = nn.ModuleList([
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                nn.Dropout2d(0.1)
            ),
            self._add_balanced_attention(resnet.layer1, 256),
            self._add_balanced_attention(resnet.layer2, 512),
            self._add_balanced_attention(resnet.layer3, 1024),
            self._add_balanced_attention(resnet.layer4, 2048)
        ])

        # Add transformer block for better feature extraction
        self.transformer = TransformerBlock(2048)
        
        # Class-specific features with adjusted dimensions
        self.class_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2)
            ) for _ in range(num_classes)
        ])
        
        # Improved classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * num_classes, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
        
    def _add_balanced_attention(self, layer, channels):
        return nn.Sequential(
            layer,
            nn.BatchNorm2d(channels),
            SpatialAttention(),
            ChannelAttention(channels),
            nn.GroupNorm(8, channels)
        )
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_attention=False):
        features = x
        attention_maps = []
        
        # Forward through encoder
        for layer in self.encoder:
            features = layer(features)
            if isinstance(layer[-2], ChannelAttention):
                attention_maps.append(layer[-2].last_attention)
        
        # Apply transformer
        features = self.transformer(features)
        
        # Class-specific feature extraction
        class_features = []
        for class_extractor in self.class_features:
            class_feat = class_extractor(features)
            class_features.append(class_feat)
        
        # Combine features
        combined = torch.cat(class_features, dim=1)
        
        # Final classification
        outputs = self.cls_head(combined)
        
        if return_attention:
            return outputs, attention_maps[-1] if attention_maps else None
        return outputs