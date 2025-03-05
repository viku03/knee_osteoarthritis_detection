import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import math
import torch.utils.checkpoint

class IntegratedHiCAXNet(nn.Module):
    def __init__(self, num_classes=5, use_gradient_checkpointing=False):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Initialize backbone and encoders
        self.anatomical_encoder = self._create_anatomical_encoder()
        self.global_encoder = self._create_global_encoder()
        self.local_encoder = self._create_local_encoder()
        
        # Cross attention and routing
        self.cross_attention = self._create_cross_attention()
        self.routing = self._create_routing_module()
        
        # Final classifier with improved regularization
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )
        
        self._initialize_weights()
        
    def _create_anatomical_encoder(self):
        return nn.Sequential(
            nn.Conv2d(2, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
    def _create_global_encoder(self):
        resnet = resnet18(pretrained=True)
        # Modify first conv for grayscale while preserving weights
        pretrained_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.data = pretrained_conv1.weight.data.sum(
                dim=1, keepdim=True
            ) / 3.0
        return nn.Sequential(*list(resnet.children())[:-2])
        
    def _create_local_encoder(self):
        return LocalDetailEncoder()
        
    def _create_cross_attention(self):
        return CrossScaleAttention()
        
    def _create_routing_module(self):
        return DynamicRoutingModule(in_channels=768)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        if self.use_gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x)
        return self._forward_normal(x)
        
    def _forward_normal(self, x):
        coordinates = self.generate_coordinate_grid(x)
        anatomical_features = self.anatomical_encoder(coordinates)
        global_features = self.global_encoder(x)
        local_features = self.local_encoder(x)
        
        fused_features = self.cross_attention(
            global_features, local_features, anatomical_features)
        routed_features = self.routing(fused_features)
        return self.classifier(routed_features)
        
    def _forward_with_checkpointing(self, x):
        coordinates = self.generate_coordinate_grid(x)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
            
        anatomical_features = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.anatomical_encoder), coordinates)
        global_features = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.global_encoder), x)
        local_features = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.local_encoder), x)
        fused_features = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.cross_attention), 
            global_features, local_features, anatomical_features)
        routed_features = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.routing), fused_features)
        output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.classifier), routed_features)
            
        return output
        
    def generate_coordinate_grid(self, x):
        batch_size, _, h, w = x.size()
        grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        coordinates = torch.stack([grid_h, grid_w], dim=0)
        return coordinates.unsqueeze(0).repeat(batch_size, 1, 1, 1)

class LocalDetailEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        channels = [in_channels, base_channels, base_channels * 2, 
                   base_channels * 4, base_channels * 8]
                   
        for i in range(len(channels) - 1):
            self.encoders.append(self._make_encoder_block(
                channels[i], channels[i + 1]))
            self.skip_connections.append(
                nn.Conv2d(channels[i + 1], channels[-1], 1))
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        features = []
        for encoder, skip in zip(self.encoders, self.skip_connections):
            x = encoder(x)
            skip_out = skip(x)
            features.append(skip_out)
            
        target_size = features[-1].shape[-2:]
        aligned_features = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, 
                                  mode='bilinear', align_corners=False)
            aligned_features.append(feat)
            
        return torch.sum(torch.stack(aligned_features, dim=0), dim=0)

class AnatomicalPriorEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.norm1 = nn.BatchNorm2d(hidden_dim)  # Changed from LayerNorm to BatchNorm2d
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(hidden_dim)  # Changed from LayerNorm to BatchNorm2d
        
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x

class GlobalContextEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        # Modified ResNet backbone
        resnet = resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global context module
        self.context_module = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, groups=32),
            nn.BatchNorm2d(512),  # Changed from LayerNorm to BatchNorm2d
            nn.ReLU(),
            nn.Conv2d(512, 512, 1)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.context_module(x)
        return x

class CrossScaleAttention(nn.Module):
    def __init__(self, global_dim=512, local_dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (global_dim // num_heads) ** -0.5
        
        # Projections
        self.global_proj = nn.Conv2d(global_dim, global_dim, 1)
        self.local_proj = nn.Conv2d(local_dim, global_dim, 1)
        self.anatomical_proj = nn.Conv2d(64, global_dim, 1)
        
        # Attention projections
        self.to_queries = nn.Conv2d(global_dim, global_dim, 1)
        self.to_keys = nn.Conv2d(global_dim, global_dim, 1)
        self.to_values = nn.Conv2d(global_dim, global_dim, 1)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(global_dim * 2, global_dim, 1),
            nn.BatchNorm2d(global_dim),
            nn.ReLU(),
            nn.Conv2d(global_dim, 768, 1)
        )
        
    def forward(self, global_features, local_features, anatomical_features):
        b, _, h, w = global_features.shape
        
        # Resize features
        local_features = F.interpolate(local_features, size=(h, w), 
                                     mode='bilinear', align_corners=False)
        anatomical_features = F.interpolate(anatomical_features, size=(h, w),
                                          mode='bilinear', align_corners=False)
        
        # Project features
        global_features = self.global_proj(global_features)
        local_features = self.local_proj(local_features)
        anatomical_features = self.anatomical_proj(anatomical_features)
        
        # Combine local and anatomical features
        local_features = local_features + anatomical_features
        
        # Multi-head attention
        queries = self.to_queries(global_features)
        keys = self.to_keys(local_features)
        values = self.to_values(local_features)
        
        # Reshape for attention
        queries = queries.view(b, self.num_heads, -1, h * w)
        keys = keys.view(b, self.num_heads, -1, h * w)
        values = values.view(b, self.num_heads, -1, h * w)
        
        # Compute attention
        attn = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attended = torch.matmul(attn, values).view(b, -1, h, w)
        
        # Final fusion
        concat_features = torch.cat([attended, global_features], dim=1)
        fused = self.feature_fusion(concat_features)
        return F.adaptive_avg_pool2d(fused, (1, 1)).squeeze(-1).squeeze(-1)

class DynamicRoutingModule(nn.Module):
    def __init__(self, in_channels, num_experts=4, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.diversity_coef = 0.2
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, in_channels)
            ) for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Get routing weights
        logits = self.router(x) / self.temperature
        routing_weights = F.softmax(logits, dim=1)
        
        # Calculate diversity loss
        entropy = -(routing_weights * torch.log(routing_weights + 1e-6)).sum(dim=1).mean()
        uniform = torch.ones_like(routing_weights) / self.num_experts
        self.diversity_loss = -entropy + self.diversity_coef * F.kl_div(
            routing_weights.log(), uniform, reduction='batchmean')
        
        # Route through experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x) + x  # Residual connection
            weighted_out = expert_out * routing_weights[:, i:i+1]
            expert_outputs.append(weighted_out)
        
        return torch.stack(expert_outputs).sum(dim=0)

    
class AdamWGC(torch.optim.AdamW):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if len(p.shape) > 1:
                    # More numerically stable mean computation
                    dims = tuple(range(1, len(p.shape)))
                    grad.sub_(grad.mean(dim=dims, keepdim=True))
                    
        return super().step(closure)
    
def create_model_and_optimizer(config):
    model = IntegratedHiCAXNet(
        num_classes=config['num_classes'],
        use_gradient_checkpointing=config['memory_management']['gradient_checkpointing']
    )
    
    # More comprehensive parameter grouping
    param_groups = {
        'decay': {'params': [], 'weight_decay': config['weight_decay']},
        'no_decay': {'params': [], 'weight_decay': 0.0}
    }
    
    for name, param in model.named_parameters():
        if any(term in name for term in ['bias', 'norm', 'temperature']):
            param_groups['no_decay']['params'].append(param)
        else:
            param_groups['decay']['params'].append(param)
    
    optimizer = AdamWGC(list(param_groups.values()), 
                       lr=config['base_learning_rate'])
    
    criterion = BalancedFocalLoss(
        num_classes=config['num_classes'],
        gamma=config['focal_loss_params']['gamma'],
        smoothing=config['focal_loss_params']['smoothing'],
    )
    
    if config['memory_management']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        
    return model, optimizer, criterion
        
class BalancedFocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=0.5, smoothing=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.eps = 1e-7
        
    def forward(self, pred, target):
        # Convert target to one-hot with higher precision
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        
        # Apply label smoothing
        target_smooth = (1 - self.smoothing) * target_one_hot + \
                       self.smoothing / self.num_classes
        
        # Compute probabilities with better numerical stability
        pred_softmax = F.softmax(pred, dim=1).clamp(min=self.eps, max=1-self.eps)
        pred_logsoft = F.log_softmax(pred, dim=1)
        
        # Compute focal weights
        pt = torch.sum(target_smooth * pred_softmax, dim=1)
        focal_weight = ((1 - pt) + self.eps).pow(self.gamma)
        
        # Compute loss with improved stability
        loss = -torch.sum(target_smooth * pred_logsoft, dim=1)
        focal_loss = focal_weight * loss
        
        return focal_loss.mean()