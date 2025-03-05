import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet50

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm([dim])  # Changed to handle 4D input
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm([dim])  # Changed to handle 4D input
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape for attention: (B, C, H, W) -> (H*W, B, C)
        x_flat = x.flatten(2).permute(2, 0, 1)
        
        # Apply attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        
        # First residual connection and norm
        x_flat = x_flat + attn_out
        x_flat = self.norm1(x_flat.permute(1, 0, 2)).permute(1, 0, 2)
        
        # MLP and second residual
        mlp_out = self.mlp(x_flat)
        x_flat = x_flat + mlp_out
        x_flat = self.norm2(x_flat.permute(1, 0, 2)).permute(1, 0, 2)
        
        # Reshape back: (H*W, B, C) -> (B, C, H, W)
        return x_flat.permute(1, 2, 0).view(B, C, H, W)

class HybridModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.3, stochastic_depth_prob=0.1):
        super().__init__()
        # Initialize ResNet50 backbone with efficient settings
        resnet = resnet50(weights='IMAGENET1K_V1')
        
        # Freeze first two layers to prevent overfitting
        for param in list(resnet.parameters())[:5]:
            param.requires_grad = False
            
        # Freeze BatchNorm layers for transfer learning
        for module in resnet.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
        
        # Enhanced encoder with residual attention
        self.encoder = nn.ModuleList([
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                nn.Dropout2d(0.1)
            ),
            self._add_attention(resnet.layer1),
            self._add_attention(resnet.layer2),
            self._add_attention(resnet.layer3),
            self._add_attention(resnet.layer4)
        ])

        # Progressive dropout with noise
        self.dropout1 = nn.Sequential(
            nn.Dropout(dropout_rate),
            GaussianNoise(0.01)
        )
        self.dropout2 = nn.Sequential(
            nn.Dropout(dropout_rate * 1.5),
            GaussianNoise(0.02)
        )
        
        # Enhanced transformer block with stochastic depth
        # In HybridModel.__init__
        self.transformer = nn.Sequential(
            TransformerBlock(2048, num_heads=8, mlp_ratio=4),
            StochasticDepth(stochastic_depth_prob, mode='batch')
        )
        
        # Decoder with attention and residual connections
        decoder_channels = [2048, 1024, 512, 256, 128]
        self.decoder = nn.ModuleList([
            self._make_decoder_block(in_ch, out_ch, dropout_rate)
            for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:])
        ])
        
        # Enhanced segmentation head with deep supervision
        self.seg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch//2, kernel_size=3, padding=1),
                nn.GroupNorm(8, ch//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch//2, 1, kernel_size=1)
            ) for ch in decoder_channels[1:]
        ])
        
        # Enhanced classification head with auxiliary supervision
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            self.dropout1,
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            self.dropout2,
            nn.Linear(512, num_classes)
        )
        
        # Auxiliary classification heads for deep supervision
        self.aux_cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch, num_classes)
            ) for ch in [1024, 512, 256]
        ])

        self._initialize_weights()
        self.use_checkpointing = True
        
    def _add_attention(self, layer):
        """Add spatial and channel attention to ResNet layer"""
        return nn.Sequential(
            layer,
            SpatialAttention(),
            ChannelAttention(layer[-1].conv3.out_channels)
        )
        
    def _make_decoder_block(self, in_channels, out_channels, dropout_rate):
        """Create decoder block with advanced features"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(8, out_channels),  # Group norm instead of batch norm
            SqueezeExcitation(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def _initialize_weights(self):
        """Advanced weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x, return_attention=False):
        # Lists to store features for skip connections and deep supervision
        encoder_features = []
        aux_cls_outputs = []
        seg_outputs = []
        
        # Encode with memory efficient attention
        features = x
        for idx, layer in enumerate(self.encoder):
            features = layer(features)
            if idx > 0:  # Skip first layer features
                encoder_features.append(features)
        
        # Apply transformer with optional attention return
        if return_attention:
            transformed, attention_maps = self._transform_with_attention(features)
        else:
            transformed = torch.utils.checkpoint.checkpoint(
                self.transformer,
                features
            ) if self.training else self.transformer(features)
        
        # Decode with skip connections and deep supervision
        decoded = transformed
        for idx, (decoder_layer, seg_head) in enumerate(zip(self.decoder, self.seg_heads)):
            # Add skip connection
            if idx < len(encoder_features):
                skip_features = encoder_features[-(idx+1)]
                if decoded.shape[2:] != skip_features.shape[2:]:
                    decoded = F.interpolate(decoded, size=skip_features.shape[2:],
                                         mode='bilinear', align_corners=True)
                decoded = decoded + skip_features
            
            # Apply decoder layer
            decoded = decoder_layer(decoded)
            
            # Generate auxiliary outputs
            if idx < len(self.aux_cls_heads):
                aux_cls_outputs.append(self.aux_cls_heads[idx](decoded))
            
            # Generate segmentation output
            seg_out = seg_head(decoded)
            if seg_out.shape[2:] != x.shape[2:]:
                seg_out = F.interpolate(seg_out, size=x.shape[2:],
                                     mode='bilinear', align_corners=True)
            seg_outputs.append(seg_out)
        
        # Main classification output
        cls_out = self.cls_head(encoder_features[-1])
        
        if self.training:
            # During training, return all outputs for deep supervision
            return (
                seg_outputs[-1],  # Main segmentation output
                cls_out,  # Main classification output
                seg_outputs[:-1],  # Auxiliary segmentation outputs
                aux_cls_outputs,  # Auxiliary classification outputs
                attention_maps if return_attention else None
            )
        else:
            # During inference, return only main outputs
            return (
                seg_outputs[-1],
                cls_out,
                attention_maps if return_attention else None
            )

    def forward(self, x, return_attention=False):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x,
                return_attention,
                preserve_rng_state=True
            )
        return self._forward_impl(x, return_attention)

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
        
    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = (avg_out + max_out).view(b, c, 1, 1)
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
    """
    Implements Stochastic Depth regularization as described in 
    "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382)
    """
    def __init__(self, prob, mode='batch'):
        """
        Args:
            prob (float): probability of dropping the path
            mode (str): 'batch' or 'row'. In 'batch' mode, the same channels are dropped for each sample in the batch.
        """
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

        # Generate random tensor with probability of keeping the path
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) > self.prob
        random_tensor = random_tensor.to(x.dtype) / (1 - self.prob)
        
        return x * random_tensor