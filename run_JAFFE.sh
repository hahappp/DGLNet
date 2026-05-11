# Model/model.py
import torch
import torch.nn as nn
from .layers import LayerNorm
from .modules import FGCBlock, LDF, UltimateBlock

class ConvNeXt_Streamlined(nn.Module):
    """
    ConvNeXt_Streamlined: The complete DGLNet Model Architecture.
    Leverages progressive feature extraction, mid-stage global context prior guidance,
    late-stage dynamic receptive-field scaling, and efficient downsampled fusion.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000,
                 depths: list = None, dims: list = None,
                 drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()

        if depths is None: depths = [3, 3, 9, 3]
        if dims is None: dims = [96, 192, 384, 768]

        # Downsample Layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Stage 1: Standard
        stage1 = nn.Sequential(*[
            UltimateBlock(dim=dims[0], drop_rate=dp_rates[cur + j],
                          layer_scale_init_value=layer_scale_init_value,
                          block_type='standard')
            for j in range(depths[0])])
        self.stages.append(stage1)
        cur += depths[0]

        # Stage 2: Standard
        stage2 = nn.Sequential(*[
            UltimateBlock(dim=dims[1], drop_rate=dp_rates[cur + j],
                          layer_scale_init_value=layer_scale_init_value,
                          block_type='standard')
            for j in range(depths[1])])
        self.stages.append(stage2)
        cur += depths[1]

        # FGC Connector (FGCBlock) between Stage 2 and Stage 3
        self.global_connector = FGCBlock(in_channels=dims[1], reduction=16)

        # Stage 3: DSK-Enhanced
        stage3 = nn.Sequential(*[
            UltimateBlock(dim=dims[2], drop_rate=dp_rates[cur + j],
                          layer_scale_init_value=layer_scale_init_value,
                          block_type='dsk')
            for j in range(depths[2])])
        self.stages.append(stage3)
        cur += depths[2]

        # Stage 4: Standard Lite (MLP ratio compressed to 2.0 to avoid redundancy)
        stage4 = nn.Sequential(*[
            UltimateBlock(dim=dims[3], drop_rate=dp_rates[cur + j],
                          layer_scale_init_value=layer_scale_init_value,
                          mlp_ratio=2.0,
                          block_type='standard')
            for j in range(depths[3])])
        self.stages.append(stage4)

        # Feature Fusion & Classification Head
        # Downsample fusion structure via LDF: transforms feat_s3 (14x14) and feat_s4 (7x7) into 7x7 map
        self.fusion_block = LDF(dim_low=dims[2], dim_high=dims[3], out_dim=dims[3])
        self.norm = nn.LayerNorm(dims[3], eps=1e-6)
        self.head = nn.Sequential(
            nn.Linear(dims[3], 256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.apply(self._init_weights)
        self.head[-1].weight.data.mul_(head_init_scale)
        self.head[-1].bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        x = self.downsample_layers[1](x)
        x = self.stages[1](x)

        # Apply Global Connector
        x = self.global_connector(x)

        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        x_s3 = x  # 14x14 low-level dynamic features

        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        x_s4 = x  # 7x7 high-level semantic features
        return x_s3, x_s4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_s3, feat_s4 = self.forward_features(x)
        fused_map = self.fusion_block(feat_s3, feat_s4)
        fused_vec = self.norm(fused_map.mean([-2, -1]))
        output = self.head(fused_vec)
        return output


def convnext_tiny_ultimate(num_classes: int, drop_path_rate: float = 0.0):
    model = ConvNeXt_Streamlined(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )
    return model


if __name__ == '__main__':
    # Local check for pipeline validation
    num_emotions = 7
    input_tensor = torch.randn(4, 3, 224, 224)
    model = convnext_tiny_ultimate(num_classes=num_emotions, drop_path_rate=0.1)
    output = model(input_tensor)
    print(f"DGLNet Streamlined Skeleton Model initialized successfully.")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1_000_000:.2f} M")