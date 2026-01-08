# Architecture Patterns for Peripheral Vision Language Models

## Overview

This document details specific architectural patterns and design choices for implementing peripheral vision in multimodal language models. Each pattern offers different trade-offs between efficiency, accuracy, and implementation complexity.

## Pattern 1: Foveal-Peripheral Dual Stream

### Architecture

```
Input Image
    ├─→ Foveal Stream (High Resolution)
    │   ├─→ Region Extraction
    │   ├─→ High-Res CNN/ViT
    │   └─→ Detailed Features
    │
    └─→ Peripheral Stream (Low Resolution)
        ├─→ Downsampling
        ├─→ Efficient Encoder
        └─→ Context Features
            ↓
        Feature Fusion
            ↓
        Language Model
            ↓
        Output
```

### Implementation

```python
import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel

class DualStreamPeripheralVLM(nn.Module):
    """
    Dual-stream architecture with separate encoders for foveal and peripheral regions.
    """
    def __init__(
        self,
        foveal_encoder='google/vit-base-patch16-224',
        peripheral_encoder='google/vit-small-patch16-224',
        language_model='gpt2',
        foveal_size=224,
        peripheral_size=384
    ):
        super().__init__()
        
        self.foveal_size = foveal_size
        self.peripheral_size = peripheral_size
        
        # High-capacity encoder for foveal region
        self.foveal_encoder = ViTModel.from_pretrained(foveal_encoder)
        
        # Lightweight encoder for peripheral vision
        self.peripheral_encoder = ViTModel.from_pretrained(peripheral_encoder)
        
        # Projection layers
        foveal_dim = self.foveal_encoder.config.hidden_size
        peripheral_dim = self.peripheral_encoder.config.hidden_size
        
        self.foveal_proj = nn.Linear(foveal_dim, 768)
        self.peripheral_proj = nn.Linear(peripheral_dim, 768)
        
        # Cross-attention for fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # Language model
        self.language_model = GPT2LMHeadModel.from_pretrained(language_model)
        
        # Adapter to connect vision to language
        self.vision_to_language = nn.Linear(768, self.language_model.config.n_embd)
        
    def extract_foveal_region(self, image, focus_point=None):
        """Extract high-resolution foveal region."""
        if focus_point is None:
            # Default to center
            focus_point = (image.size(2) // 2, image.size(3) // 2)
        
        h, w = focus_point
        half_size = self.foveal_size // 2
        
        h_start = max(0, min(image.size(2) - self.foveal_size, h - half_size))
        w_start = max(0, min(image.size(3) - self.foveal_size, w - half_size))
        
        foveal_region = image[
            :, :,
            h_start:h_start + self.foveal_size,
            w_start:w_start + self.foveal_size
        ]
        
        return foveal_region, (h_start, w_start)
    
    def encode_visual(self, image, focus_point=None):
        """Encode image using dual-stream approach."""
        batch_size = image.size(0)
        
        # Foveal stream: high-resolution detail
        foveal_region, focus_coords = self.extract_foveal_region(image, focus_point)
        foveal_outputs = self.foveal_encoder(pixel_values=foveal_region)
        foveal_features = self.foveal_proj(foveal_outputs.last_hidden_state)
        
        # Peripheral stream: low-resolution context
        peripheral_image = torch.nn.functional.interpolate(
            image,
            size=(self.peripheral_size, self.peripheral_size),
            mode='bilinear',
            align_corners=False
        )
        peripheral_outputs = self.peripheral_encoder(pixel_values=peripheral_image)
        peripheral_features = self.peripheral_proj(peripheral_outputs.last_hidden_state)
        
        # Fuse features using cross-attention
        # Foveal features attend to peripheral context
        fused_features, _ = self.fusion(
            foveal_features,
            peripheral_features,
            peripheral_features
        )
        
        return fused_features
    
    def forward(self, image, input_ids, focus_point=None):
        """Forward pass through dual-stream VLM."""
        # Encode visual input
        visual_features = self.encode_visual(image, focus_point)
        
        # Project to language model dimension
        visual_embeds = self.vision_to_language(visual_features)
        
        # Get text embeddings
        text_embeds = self.language_model.transformer.wte(input_ids)
        
        # Concatenate visual and text embeddings
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # Generate output
        outputs = self.language_model(inputs_embeds=combined_embeds)
        
        return outputs
```

### Advantages
- Clear separation of concerns
- Easy to optimize each stream independently
- Flexible encoder choices
- Good balance of efficiency and accuracy

### Disadvantages
- Requires training two separate encoders
- Fixed foveal region size
- May miss important details outside foveal region

## Pattern 2: Attention-Weighted Multi-Scale

### Architecture

```
Input Image
    ↓
Multi-Scale Pyramid
    ├─→ Scale 1 (Full Resolution)
    ├─→ Scale 2 (1/2 Resolution)
    ├─→ Scale 3 (1/4 Resolution)
    └─→ Scale 4 (1/8 Resolution)
        ↓
Query-Guided Attention
        ↓
Weighted Feature Aggregation
        ↓
Language Model
        ↓
Output
```

### Implementation

```python
class AttentionWeightedMultiScaleVLM(nn.Module):
    """
    Multi-scale processing with query-guided attention weighting.
    """
    def __init__(
        self,
        scales=[1.0, 0.5, 0.25, 0.125],
        base_channels=64,
        attention_heads=8
    ):
        super().__init__()
        
        self.scales = scales
        
        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList([
            self._make_encoder(base_channels * (2 ** i))
            for i in range(len(scales))
        ])
        
        # Query encoder for attention
        self.query_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=2
        )
        
        # Scale attention module
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=attention_heads,
            batch_first=True
        )
        
        # Feature aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(512 * len(scales), 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _make_encoder(self, channels):
        """Create a scale-specific encoder."""
        return nn.Sequential(
            nn.Conv2d(3, channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(channels * 2 * 7 * 7, 512)
        )
    
    def encode_multiscale(self, image):
        """Encode image at multiple scales."""
        scale_features = []
        
        for scale, encoder in zip(self.scales, self.scale_encoders):
            if scale < 1.0:
                # Downsample image
                h, w = image.size(2), image.size(3)
                scaled_h, scaled_w = int(h * scale), int(w * scale)
                scaled_image = torch.nn.functional.interpolate(
                    image,
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_image = image
            
            features = encoder(scaled_image)
            scale_features.append(features)
        
        return torch.stack(scale_features, dim=1)  # [B, num_scales, 512]
    
    def compute_scale_attention(self, scale_features, query_embedding):
        """Compute attention weights over scales based on query."""
        batch_size = scale_features.size(0)
        
        # Expand query to match number of scales
        query = query_embedding.unsqueeze(1).expand(-1, len(self.scales), -1)
        
        # Compute attention
        attended_features, attention_weights = self.scale_attention(
            query,
            scale_features,
            scale_features
        )
        
        return attended_features, attention_weights
    
    def forward(self, image, query_embedding):
        """Forward pass with query-guided multi-scale attention."""
        # Encode at multiple scales
        scale_features = self.encode_multiscale(image)
        
        # Encode query
        query_embed = self.query_encoder(query_embedding.unsqueeze(1))
        
        # Compute scale attention
        attended_features, attention_weights = self.compute_scale_attention(
            scale_features,
            query_embed.squeeze(1)
        )
        
        # Aggregate features
        flattened = attended_features.flatten(start_dim=1)
        aggregated = self.aggregator(flattened)
        
        return aggregated, attention_weights
```

### Advantages
- Adaptive to query and content
- No explicit foveal region selection needed
- Learns optimal scale weighting
- Handles multi-scale objects well

### Disadvantages
- Higher computational cost (multiple scales)
- More complex training
- May over-allocate to certain scales

## Pattern 3: Cascaded Refinement

### Architecture

```
Stage 1: Coarse (Full Image, Low Res)
    ↓
Region Selection
    ↓
Stage 2: Medium (Selected Regions, Med Res)
    ↓
Refinement Selection
    ↓
Stage 3: Fine (Refined Regions, High Res)
    ↓
Hierarchical Fusion
    ↓
Output
```

### Implementation

```python
class CascadedRefinementVLM(nn.Module):
    """
    Cascaded refinement with progressive resolution increase.
    """
    def __init__(
        self,
        stage_resolutions=[64, 128, 256],
        num_regions_per_stage=[1, 3, 9],
        hidden_dim=512
    ):
        super().__init__()
        
        self.stage_resolutions = stage_resolutions
        self.num_regions = num_regions_per_stage
        self.num_stages = len(stage_resolutions)
        
        # Stage encoders with increasing capacity
        self.stage_encoders = nn.ModuleList([
            self._make_stage_encoder(res, hidden_dim * (i + 1))
            for i, res in enumerate(stage_resolutions)
        ])
        
        # Region proposal networks
        self.region_proposers = nn.ModuleList([
            RegionProposalNetwork(hidden_dim * (i + 1), num_regions_per_stage[i + 1])
            for i in range(self.num_stages - 1)
        ])
        
        # Hierarchical fusion
        self.fusion = HierarchicalFusion(
            input_dims=[hidden_dim * (i + 1) for i in range(self.num_stages)],
            output_dim=hidden_dim
        )
        
    def _make_stage_encoder(self, resolution, hidden_dim):
        """Create encoder for specific stage."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, hidden_dim)
        )
    
    def forward(self, image, query=None):
        """Forward pass through cascaded stages."""
        batch_size = image.size(0)
        
        stage_features = []
        current_regions = [None]  # Full image for stage 1
        
        for stage_idx in range(self.num_stages):
            resolution = self.stage_resolutions[stage_idx]
            
            # Process regions at current stage
            stage_outputs = []
            for region in current_regions:
                if region is None:
                    # Full image at low resolution
                    stage_input = torch.nn.functional.interpolate(
                        image,
                        size=(resolution, resolution),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    # Extract and resize region
                    stage_input = self._extract_region(image, region, resolution)
                
                output = self.stage_encoders[stage_idx](stage_input)
                stage_outputs.append(output)
            
            # Aggregate outputs from all regions at this stage
            stage_feature = torch.stack(stage_outputs, dim=1).mean(dim=1)
            stage_features.append(stage_feature)
            
            # Propose regions for next stage
            if stage_idx < self.num_stages - 1:
                current_regions = self.region_proposers[stage_idx](
                    stage_feature,
                    query
                )
        
        # Fuse features from all stages
        fused_features = self.fusion(stage_features)
        
        return fused_features
    
    def _extract_region(self, image, region, target_size):
        """Extract and resize a specific region from image."""
        x1, y1, x2, y2 = region
        region_img = image[:, :, y1:y2, x1:x2]
        
        resized = torch.nn.functional.interpolate(
            region_img,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
        
        return resized


class RegionProposalNetwork(nn.Module):
    """Proposes regions of interest for next stage."""
    def __init__(self, input_dim, num_regions):
        super().__init__()
        self.num_regions = num_regions
        
        self.proposal_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_regions * 4)  # x1, y1, x2, y2 for each region
        )
    
    def forward(self, features, query=None):
        """Generate region proposals."""
        proposals = self.proposal_head(features)
        proposals = proposals.view(-1, self.num_regions, 4)
        
        # Apply sigmoid to get normalized coordinates
        proposals = torch.sigmoid(proposals)
        
        return proposals


class HierarchicalFusion(nn.Module):
    """Fuse features from multiple hierarchical stages."""
    def __init__(self, input_dims, output_dim):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim)
            for dim in input_dims
        ])
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(self, stage_features):
        """Fuse features from all stages."""
        # Project all features to same dimension
        projected = [
            proj(feat).unsqueeze(1)
            for proj, feat in zip(self.projections, stage_features)
        ]
        
        # Stack features
        stacked = torch.cat(projected, dim=1)  # [B, num_stages, output_dim]
        
        # Self-attention over stages
        fused, _ = self.fusion_attention(stacked, stacked, stacked)
        
        # Aggregate (mean pooling over stages)
        aggregated = fused.mean(dim=1)
        
        # Final projection
        output = self.output_proj(aggregated)
        
        return output
```

### Advantages
- Progressive refinement for efficiency
- Natural coarse-to-fine processing
- Can focus computation on important regions
- Mimics human visual search behavior

### Disadvantages
- Sequential processing (cannot parallelize stages)
- Region proposal adds complexity
- Potential error propagation across stages
- More difficult to train end-to-end

## Pattern 4: Learnable Foveation with Reinforcement Learning

### Architecture

```
Image Input
    ↓
Foveation Policy Network (RL Agent)
    ↓
Sample Focus Points
    ↓
Extract Foveal Patches
    ↓
Process with High-Res Encoder
    ↓
Combine with Peripheral Context
    ↓
Task Network (VQA, Captioning, etc.)
    ↓
Reward = Task Performance + Efficiency Bonus
    ↓
Update Policy
```

### Implementation

```python
import torch.distributions as dist

class RLFoveatedVLM(nn.Module):
    """
    Reinforcement learning-based foveation policy.
    """
    def __init__(
        self,
        num_glimpses=3,
        foveal_size=128,
        hidden_dim=512,
        action_dim=2  # (x, y) coordinates
    ):
        super().__init__()
        
        self.num_glimpses = num_glimpses
        self.foveal_size = foveal_size
        
        # Peripheral encoder for full image context
        self.peripheral_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, hidden_dim)
        )
        
        # Foveal encoder for high-res patches
        self.foveal_encoder = nn.Sequential(
            nn.Conv2d(3, 128, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, hidden_dim)
        )
        
        # Policy network (Actor)
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 256),  # context + previous glimpse
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Location head (outputs mean and std for Gaussian policy)
        self.location_mean = nn.Linear(128, action_dim)
        self.location_std = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softplus()  # Ensure positive std
        )
        
        # Value network (Critic)
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Task head (e.g., VQA)
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim * (self.num_glimpses + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1000)  # Example: 1000-way classification
        )
    
    def sample_location(self, context_features, previous_glimpse_features):
        """Sample next glimpse location from policy."""
        # Concatenate context and previous glimpse
        policy_input = torch.cat([context_features, previous_glimpse_features], dim=1)
        
        # Compute policy parameters
        policy_features = self.policy_network(policy_input)
        loc_mean = self.location_mean(policy_features)
        loc_std = self.location_std(policy_features)
        
        # Create Gaussian distribution
        location_dist = dist.Normal(loc_mean, loc_std)
        
        # Sample location
        location = location_dist.rsample()  # Reparameterization trick
        log_prob = location_dist.log_prob(location).sum(dim=-1)
        
        # Normalize to [0, 1]
        location = torch.sigmoid(location)
        
        return location, log_prob
    
    def extract_glimpse(self, image, location):
        """Extract foveal glimpse at specified location."""
        batch_size = image.size(0)
        h, w = image.size(2), image.size(3)
        
        # Convert normalized location to pixel coordinates
        center_h = (location[:, 0] * h).long()
        center_w = (location[:, 1] * w).long()
        
        glimpses = []
        for i in range(batch_size):
            h_start = max(0, center_h[i] - self.foveal_size // 2)
            w_start = max(0, center_w[i] - self.foveal_size // 2)
            h_end = min(h, h_start + self.foveal_size)
            w_end = min(w, w_start + self.foveal_size)
            
            glimpse = image[i:i+1, :, h_start:h_end, w_start:w_end]
            
            # Pad if necessary
            if glimpse.size(2) < self.foveal_size or glimpse.size(3) < self.foveal_size:
                glimpse = torch.nn.functional.pad(
                    glimpse,
                    (0, self.foveal_size - glimpse.size(3),
                     0, self.foveal_size - glimpse.size(2))
                )
            
            glimpses.append(glimpse)
        
        return torch.cat(glimpses, dim=0)
    
    def forward(self, image, training=True):
        """Forward pass with RL-based foveation."""
        batch_size = image.size(0)
        
        # Encode peripheral context
        context_features = self.peripheral_encoder(
            torch.nn.functional.interpolate(image, size=(128, 128))
        )
        
        # Initialize
        glimpse_features_list = []
        log_probs = []
        locations = []
        
        previous_glimpse_features = torch.zeros_like(context_features)
        
        # Take multiple glimpses
        for glimpse_idx in range(self.num_glimpses):
            # Sample location
            if training:
                location, log_prob = self.sample_location(
                    context_features,
                    previous_glimpse_features
                )
            else:
                # Greedy selection during inference
                with torch.no_grad():
                    policy_input = torch.cat(
                        [context_features, previous_glimpse_features],
                        dim=1
                    )
                    policy_features = self.policy_network(policy_input)
                    location = torch.sigmoid(self.location_mean(policy_features))
                    log_prob = None
            
            locations.append(location)
            if log_prob is not None:
                log_probs.append(log_prob)
            
            # Extract and encode glimpse
            glimpse = self.extract_glimpse(image, location)
            glimpse_features = self.foveal_encoder(glimpse)
            glimpse_features_list.append(glimpse_features)
            
            # Update for next iteration
            previous_glimpse_features = glimpse_features
        
        # Combine all features
        all_features = torch.cat(
            [context_features] + glimpse_features_list,
            dim=1
        )
        
        # Task prediction
        task_output = self.task_head(all_features)
        
        # Value estimation
        value = self.value_network(context_features)
        
        return {
            'task_output': task_output,
            'value': value,
            'log_probs': torch.stack(log_probs) if log_probs else None,
            'locations': torch.stack(locations),
            'context_features': context_features,
            'glimpse_features': glimpse_features_list
        }
    
    def compute_rl_loss(self, outputs, rewards, gamma=0.99):
        """Compute policy gradient loss (REINFORCE or A2C)."""
        log_probs = outputs['log_probs']
        values = outputs['value']
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(values.device)
        
        # Compute advantages
        advantages = returns - values.squeeze()
        
        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values.squeeze(), returns)
        
        return policy_loss + 0.5 * value_loss


# Training loop example
def train_rl_foveation(model, dataloader, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch['image']
            labels = batch['label']
            
            # Forward pass
            outputs = model(images, training=True)
            
            # Task loss
            task_loss = nn.functional.cross_entropy(
                outputs['task_output'],
                labels
            )
            
            # Reward = task accuracy - efficiency penalty
            with torch.no_grad():
                predictions = outputs['task_output'].argmax(dim=1)
                task_reward = (predictions == labels).float()
                efficiency_bonus = 0.1  # Bonus for using fewer glimpses
                rewards = [task_reward + efficiency_bonus] * model.num_glimpses
            
            # RL loss
            rl_loss = model.compute_rl_loss(outputs, rewards)
            
            # Total loss
            total_loss = task_loss + 0.1 * rl_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### Advantages
- Learns optimal foveation strategy from data
- Adapts to specific tasks and datasets
- Can discover non-obvious attention patterns
- Explicit efficiency-accuracy trade-off

### Disadvantages
- Training instability (RL challenges)
- Requires careful reward shaping
- Slower convergence
- More complex implementation

## Comparison of Patterns

| Pattern | Efficiency | Accuracy | Training Complexity | Flexibility | Use Cases |
|---------|-----------|----------|-------------------|-------------|-----------|
| Dual Stream | High | Medium-High | Low | Medium | General purpose, fast inference |
| Multi-Scale | Medium | High | Medium | High | Multi-scale objects, variable scene complexity |
| Cascaded | High | High | Medium-High | Medium | Sequential refinement, resource-constrained |
| RL-Based | Very High | Medium | Very High | Very High | Adaptive tasks, research applications |

## Implementation Recommendations

### For Production Systems
- **Use**: Dual Stream or Multi-Scale
- **Why**: Stable training, predictable behavior, good trade-offs

### For Research
- **Use**: RL-Based or Cascaded
- **Why**: Novel approaches, potential for breakthroughs

### For Real-Time Applications
- **Use**: Dual Stream with fixed fovea
- **Why**: Minimal latency, consistent performance

### For Maximum Accuracy
- **Use**: Multi-Scale with full resolution
- **Why**: Preserves most information, adaptive weighting

## Future Directions

1. **Hybrid Approaches**: Combine multiple patterns
2. **Neural Architecture Search**: Automated pattern discovery
3. **Meta-Learning**: Quick adaptation to new tasks
4. **Neuromorphic Implementation**: Event-based processing
5. **3D and Video**: Extend patterns to temporal domain

---

**Last Updated**: January 2025
