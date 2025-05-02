import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) 
        return x, (H, W)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  
        x = x + self.mlp(self.norm2(x))   
        return x

class SimpleViTSR(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4, num_blocks=8, upscale=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim) for _ in range(num_blocks)
        ])
        
        self.upscale = upscale
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.reconstruct = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_embed, (H, W) = self.patch_embed(x)
        x_transformed = self.transformer_blocks(x_embed)
        x_feat = x_transformed.transpose(1, 2).reshape(x.shape[0], self.embed_dim, H, W)
        x_feat_upscaled = nn.functional.interpolate(x_feat, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out = self.reconstruct(x_feat_upscaled)

        return out