# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_custom import MambaCustom as Mamba

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    RMSNorm = None


__all__ = [
    'adventurer_tiny_patch16_224', 'adventurer_small_patch16_224',
    'adventurer_base_patch16_224', 'adventurer_large_patch16_224',
]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] - patch_size[0]) // patch_size[0] + 1
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, drop_path=0.):
        super().__init__()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = SwiGLU(dim, int(dim * 2.5))
        self.norm_mlp = norm_cls(dim)

    def forward(self, hidden_states, flip=False, inference_params=None):
        x = self.norm(hidden_states)

        # Implementation details: we place two inter-layer flippings within the same block and execute
        # it every two blocks, which ensurs that the output token order of all blocks is consistent.
        x = torch.cat([x[:, :-1].flip(1), x[:, -1:]], dim=1) if flip else x

        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # heading avergae, [B, N+2, C]
        x = self.mixer(x, inference_params=inference_params)[:, 1:]  # discard the first token, [B, N+1, C]

        x = torch.cat([x[:, :-1].flip(1), x[:, -1:]], dim=1) if flip else x

        hidden_states = hidden_states + self.drop_path(x)
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm_mlp(hidden_states)))
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, drop_path=0., rms_norm=False,
    layer_idx=None, device=None, dtype=None, mamba_expand=2, mamba_d_state=64):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba,
                        d_state=mamba_d_state,
                        expand=mamba_expand,
                        layer_idx=layer_idx,
                        **ssm_cfg,
                        **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(d_model, mixer_cls, norm_cls=norm_cls, drop_path=drop_path)
   
    block.layer_idx = layer_idx
    return block

def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Adventurer(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 depth=12, 
                 embed_dim=256,
                 channels=3, 
                 num_classes=1000,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 mamba_expand=2,
                 mamba_d_state=64,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()

        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [0.0] + [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # adventurer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    layer_idx=i,
                    drop_path=dpr[i],
                    mamba_expand=mamba_expand,
                    mamba_d_state=mamba_d_state,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token        
        x = self.patch_embed(x)
        B, M, _ = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([x, cls_token], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for n, layer in enumerate(self.layers):
            x = layer(x, flip = (n // 2 == 1), inference_params=inference_params)
        x = self.norm_f(x)

        return x[:, -1]

    def forward(self, x, return_features=False, inference_params=None):
        x = self.forward_features(x, inference_params)
        if return_features:
            return x
        x = self.head(x)
        return x

@register_model
def adventurer_tiny_patch16(pretrained=False, **kwargs):
    model = Adventurer(
        patch_size=16, embed_dim=256, depth=12, rms_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def adventurer_small_patch16(pretrained=False, **kwargs):
    model = Adventurer(
        patch_size=16, embed_dim=512, depth=12, rms_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def adventurer_base_patch16(pretrained=False, **kwargs):
    model = Adventurer(
        patch_size=16, embed_dim=768, depth=12, rms_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def adventurer_base_patch8(pretrained=False, **kwargs):
    model = Adventurer(
        patch_size=8, embed_dim=768, depth=12, rms_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def adventurer_large_patch16(pretrained=False, **kwargs):
    model = Adventurer(
        patch_size=16, embed_dim=1024, depth=24, rms_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model

