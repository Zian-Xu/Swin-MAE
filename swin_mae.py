from functools import partial

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp
from utils.pos_embed import get_2d_sincos_pos_embed


class SwinMAE(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()
       
        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
       
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
           
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')
        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0) 
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int) 
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1): 
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, x):
        x = self.patch_embed(x)

        x, mask = self.window_masking(x, remove=False, mask_len_sparse=False)

        for layer in self.layers:
            x = layer(x)

        return x, mask

    def forward_decoder(self, x):
       
        x = self.first_patch_expanding(x)

        for layer in self.layers_up:
            x = layer(x)

        x = self.norm_up(x)

        x = rearrange(x, 'B H W C -> B (H W) C')

        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        loss = (loss * mask).sum() / mask.sum() 
        return loss

    def forward(self, x):
        latent, mask = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask


def swin_mae(**kwargs):
    model = SwinMAE(
        img_size=224, patch_size=4, in_chans=3,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        window_size=7, qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
