import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import math
import torch.utils.checkpoint

class EnhancedHiCAXNet(nn.Module):
    def __init__(self, num_classes=5, num_regions=4, use_gradient_checkpointing=False):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_regions = num_regions
        
        # Enhanced encoders with region-specific processing
        self.anatomical_encoder = AnatomicalPriorNetwork(num_regions=num_regions)
        self.global_encoder = EnhancedGlobalEncoder()
        self.local_encoder = RegionSpecificEncoder(num_regions=num_regions)
        
        # Biomechanical constraint module
        self.biomech_module = BiomechanicalConstraintModule()
        
        # Clinical metadata integration
        self.clinical_encoder = ClinicalMetadataEncoder()
        
        # Enhanced cross-attention with uncertainty
        self.cross_attention = UncertaintyAwareAttention()
        
        # Uncertainty-aware routing
        self.routing = UncertaintyAwareRoutingModule(in_channels=1024)
        
        # Final classifier with uncertainty estimation
        self.classifier = UncertaintyAwareClassifier(num_classes=num_classes)
        
        self._initialize_weights()
    
    def forward(self, x, clinical_data=None):
        # Generate anatomical priors and uncertainty
        anatomical_features, anatomical_uncertainty = self.anatomical_encoder(x)
        
        # Extract global and local features
        global_features = self.global_encoder(x)
        local_features = self.local_encoder(x)
        
        # Apply biomechanical constraints
        biomech_features = self.biomech_module(local_features)
        
        # Encode clinical metadata if available
        clinical_features = self.clinical_encoder(clinical_data) if clinical_data is not None else None
        
        # Cross-attention with uncertainty
        fused_features = self.cross_attention(
            global_features, local_features, anatomical_features, 
            anatomical_uncertainty, clinical_features
        )
        
        # Uncertainty-aware routing
        routed_features, routing_uncertainty = self.routing(fused_features)
        
        # Final classification with uncertainty
        logits, prediction_uncertainty = self.classifier(routed_features)
        
        return {
            'logits': logits,
            'prediction_uncertainty': prediction_uncertainty,
            'routing_uncertainty': routing_uncertainty,
            'anatomical_uncertainty': anatomical_uncertainty
        }

class AnatomicalPriorNetwork(nn.Module):
    def __init__(self, num_regions=4):
        super().__init__()
        self.num_regions = num_regions
        
        # Multi-resolution template learning
        self.template_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1, stride=2**i),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ) for i in range(3)
        ])
        
        # Learnable anatomical landmarks
        self.landmark_predictor = nn.Sequential(
            nn.Conv2d(64 * 3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_regions * 2, 1)  # x,y coordinates for each region
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(64 * 3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_regions, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Generate coordinate grid
        coordinates = self.generate_coordinate_grid(x)
        
        # Multi-resolution feature extraction
        multi_scale_features = []
        for encoder in self.template_encoder:
            features = encoder(coordinates)
            if features.shape[-2:] != x.shape[-2:]:
                features = F.interpolate(features, size=x.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            multi_scale_features.append(features)
        
        # Combine multi-scale features
        combined_features = torch.cat(multi_scale_features, dim=1)
        
        # Predict landmarks and uncertainty
        landmarks = self.landmark_predictor(combined_features)
        uncertainty = self.uncertainty_estimator(combined_features)
        
        return combined_features, uncertainty

class BiomechanicalConstraintModule(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        
        # Stress-strain pattern estimation
        self.stress_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        
        # Joint space analysis
        self.joint_analyzer = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=4),  # Region-specific
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1)
        )
        
        # Motion pathway analysis
        self.motion_analyzer = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, dilation=2)
        )
        
    def forward(self, x):
        # Estimate stress-strain patterns
        stress_patterns = self.stress_estimator(x)
        
        # Analyze joint space
        joint_features = self.joint_analyzer(stress_patterns)
        
        # Analyze motion pathways
        motion_features = self.motion_analyzer(joint_features)
        
        return torch.cat([stress_patterns, joint_features, motion_features], dim=1)

class RegionSpecificEncoder(nn.Module):
    def __init__(self, num_regions=4, in_channels=1):
        super().__init__()
        self.num_regions = num_regions
        
        # Region-specific convolutions
        self.region_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1, groups=4)
            ) for _ in range(num_regions)
        ])
        
        # Region attention
        self.region_attention = nn.Sequential(
            nn.Conv2d(64 * num_regions, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_regions, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Process each region
        region_features = []
        for conv in self.region_convs:
            region_features.append(conv(x))
        
        # Combine region features
        combined = torch.cat(region_features, dim=1)
        
        # Generate attention weights
        attention = self.region_attention(combined)
        
        # Apply attention to features
        attended_features = torch.sum(
            torch.stack([feat * att for feat, att in zip(
                region_features, attention.chunk(self.num_regions, dim=1))]),
            dim=0
        )
        
        return attended_features

class UncertaintyAwareClassifier(nn.Module):
    def __init__(self, in_features=1024, num_classes=5):
        super().__init__()
        
        # Main classification branch
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Uncertainty estimation branch
        self.uncertainty = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        logits = self.classifier(x)
        uncertainty = self.uncertainty(x)
        return logits, uncertainty

class ClinicalMetadataEncoder(nn.Module):
    def __init__(self, num_clinical_features=10):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_clinical_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
    def forward(self, clinical_data):
        if clinical_data is None:
            return None
        return self.encoder(clinical_data)

class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=5, smoothing=0.1, class_weights=None, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer('class_weights', 
                           torch.tensor(class_weights) if class_weights is not None 
                           else torch.ones(num_classes))
    
    def forward(self, pred, target, target_b=None, lam=None):
        # Convert targets to one-hot
        target_one_hot = F.one_hot(target, self.num_classes).float()
        
        # Apply mixup if provided
        if target_b is not None and lam is not None:
            target_b_one_hot = F.one_hot(target_b, self.num_classes).float()
            target_one_hot = lam * target_one_hot + (1 - lam) * target_b_one_hot
            
        # Apply label smoothing
        target_smooth = ((1 - self.smoothing) * target_one_hot + 
                        self.smoothing / self.num_classes)
        
        # Apply log softmax with better numerical stability
        log_probs = F.log_softmax(pred, dim=1)
        
        # Weight the loss by class weights
        loss = -(target_smooth * log_probs) * self.class_weights.view(1, -1)
        loss = loss.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class FocalLossWithMixup(nn.Module):
    def __init__(self, num_classes=5, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        
        # Initialize alpha for class balancing if not provided
        if alpha is None:
            self.register_buffer('alpha', torch.ones(num_classes))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
    
    def forward(self, pred, target, target_b=None, lam=None):
        # Get probabilities
        pred_softmax = F.softmax(pred, dim=1)
        
        # Convert targets to one-hot and handle mixup
        target_one_hot = F.one_hot(target, self.num_classes).float()
        if target_b is not None and lam is not None:
            target_b_one_hot = F.one_hot(target_b, self.num_classes).float()
            target_one_hot = lam * target_one_hot + (1 - lam) * target_b_one_hot
        
        # Compute focal weights
        probs_for_target = (pred_softmax * target_one_hot).sum(1)
        focal_weights = (1 - probs_for_target) ** self.gamma
        
        # Compute log probabilities with improved numerical stability
        log_probs = F.log_softmax(pred, dim=1)
        
        # Apply class weights and focal weights
        weighted_loss = -self.alpha.view(1, -1) * target_one_hot * log_probs
        focal_loss = focal_weights.view(-1, 1) * weighted_loss
        
        # Reduce loss according to reduction method
        if self.reduction == 'mean':
            return focal_loss.sum() / pred.size(0)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss.sum(1)

def get_inverse_class_weights(dataset):
    """Calculate inverse class weights for balancing"""
    label_counts = torch.bincount(torch.tensor(dataset.labels))
    total_samples = len(dataset)
    inverse_weights = total_samples / (len(label_counts) * label_counts.float())
    return inverse_weights / inverse_weights.sum()

def create_model_and_optimizer(config):
    model = EnhancedHiCAXNet(
        num_classes=config['num_classes'],
        num_regions=config['num_regions'],
        use_gradient_checkpointing=config['use_gradient_checkpointing']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    criterion = FocalLossWithMixup(
        num_classes=config['num_classes'],
        alpha=config['loss_weights']['classification'],
        beta=config['loss_weights']['anatomical'],
        gamma=config['loss_weights']['uncertainty']
    )
    
    return model, optimizer, criterion

class UncertaintyAwareAttention(nn.Module):
    def __init__(self, global_dim=512, local_dim=256, clinical_dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (global_dim // num_heads) ** -0.5
        
        # Feature projections
        self.global_proj = nn.Conv2d(global_dim, global_dim, 1)
        self.local_proj = nn.Conv2d(local_dim, global_dim, 1)
        self.anatomical_proj = nn.Conv2d(192, global_dim, 1)  # 192 = 64 * 3 from anatomical encoder
        
        # Clinical feature projection if available
        self.clinical_proj = nn.Linear(clinical_dim, global_dim)
        
        # Attention projections
        self.to_queries = nn.Conv2d(global_dim, global_dim, 1)
        self.to_keys = nn.Conv2d(global_dim, global_dim, 1)
        self.to_values = nn.Conv2d(global_dim, global_dim, 1)
        
        # Uncertainty-guided attention
        self.uncertainty_gate = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion with uncertainty
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(global_dim * 3, global_dim, 1),
            nn.BatchNorm2d(global_dim),
            nn.ReLU(),
            nn.Conv2d(global_dim, 1024, 1)  # Increased dimensionality for better feature representation
        )
        
    def forward(self, global_features, local_features, anatomical_features, 
                anatomical_uncertainty, clinical_features=None):
        b, _, h, w = global_features.shape
        
        # Resize features and apply projections
        local_features = F.interpolate(local_features, size=(h, w), 
                                     mode='bilinear', align_corners=False)
        anatomical_features = F.interpolate(anatomical_features, size=(h, w),
                                          mode='bilinear', align_corners=False)
        
        global_features = self.global_proj(global_features)
        local_features = self.local_proj(local_features)
        anatomical_features = self.anatomical_proj(anatomical_features)
        
        # Apply uncertainty gating
        uncertainty_weights = self.uncertainty_gate(anatomical_uncertainty)
        anatomical_features = anatomical_features * uncertainty_weights
        
        # Process clinical features if available
        if clinical_features is not None:
            clinical_features = self.clinical_proj(clinical_features)
            clinical_features = clinical_features.view(b, -1, 1, 1).expand(-1, -1, h, w)
            global_features = global_features + clinical_features
        
        # Multi-head attention computation
        queries = self.to_queries(global_features).view(b, self.num_heads, -1, h * w)
        keys = self.to_keys(local_features).view(b, self.num_heads, -1, h * w)
        values = self.to_values(local_features).view(b, self.num_heads, -1, h * w)
        
        # Compute attention scores
        attn = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn, values).view(b, -1, h, w)
        
        # Combine features with uncertainty-aware fusion
        concat_features = torch.cat([
            attended, 
            global_features, 
            anatomical_features
        ], dim=1)
        
        fused = self.feature_fusion(concat_features)
        return F.adaptive_avg_pool2d(fused, (1, 1)).squeeze(-1).squeeze(-1)

class UncertaintyAwareRoutingModule(nn.Module):
    def __init__(self, in_channels, num_experts=4, hidden_dim=512):
        super().__init__()
        self.num_experts = num_experts
        self.diversity_coef = 0.2
        
        # Expert networks with uncertainty estimation
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, in_channels)
            ) for _ in range(num_experts)
        ])
        
        # Expert uncertainty estimators
        self.uncertainty_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_experts)
        ])
        
        # Confidence-based router
        self.router = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Get routing weights with temperature scaling
        logits = self.router(x) / self.temperature
        routing_weights = F.softmax(logits, dim=1)
        
        # Calculate diversity loss
        entropy = -(routing_weights * torch.log(routing_weights + 1e-6)).sum(dim=1).mean()
        uniform = torch.ones_like(routing_weights) / self.num_experts
        self.diversity_loss = -entropy + self.diversity_coef * F.kl_div(
            routing_weights.log(), uniform, reduction='batchmean')
        
        # Process through experts with uncertainty estimation
        expert_outputs = []
        expert_uncertainties = []
        for i, (expert, uncertainty_estimator) in enumerate(zip(self.experts, self.uncertainty_estimators)):
            expert_out = expert(x)
            uncertainty = uncertainty_estimator(expert_out)
            
            # Apply residual connection
            expert_out = expert_out + x
            
            # Weight output by routing weight and uncertainty
            confidence = 1 - uncertainty
            weighted_out = expert_out * routing_weights[:, i:i+1] * confidence
            
            expert_outputs.append(weighted_out)
            expert_uncertainties.append(uncertainty)
        
        # Combine outputs and uncertainties
        combined_output = torch.stack(expert_outputs).sum(dim=0)
        combined_uncertainty = torch.stack(expert_uncertainties).mean(dim=0)
        
        return combined_output, combined_uncertainty

class EnhancedGlobalEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        # Modified ResNet backbone
        resnet = resnet18(pretrained=True)
        
        # Modify first conv for grayscale while preserving pretrained weights
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True) / 3.0
                
        # Extract layers before final pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Enhanced global context module
        self.context_module = nn.Sequential(
            # Multi-scale context aggregation
            nn.Conv2d(512, 512, 3, padding=1, groups=32),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Dilated convolutions for larger receptive field
            nn.Conv2d(512, 512, 3, padding=2, dilation=2, groups=32),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Channel attention
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 512, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features through backbone
        features = self.backbone(x)
        
        # Apply global context enhancement
        context = self.context_module(features)
        features = features * context
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(features)
        features = features * spatial_weights
        
        return features