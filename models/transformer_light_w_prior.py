# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class PriorEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, max_size=2000, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(max_size, num_pos_feats//2)
        self.col_embed = nn.Embedding(max_size, num_pos_feats//2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, y):
        x_emb = self.col_embed(x)
        y_emb = self.row_embed(y)
        pos = torch.cat([x_emb,y_emb], dim=-1).unsqueeze(0)
        return pos

def simple_nms(scores, nms_radius=3):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, return_attn=False):
        q = k = self.with_pos_embed(src, pos)
        src2, attn = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, need_weights=return_attn)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, return_attn=False):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, need_weights=return_attn)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attn

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, return_attn=False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn=return_attn)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, return_attn=return_attn)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            return_attn=False
            if i == n-1:
                return_attn = True
            output, attn = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, return_attn=return_attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TokenizedEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.prior_pos_embed = PriorEmbeddingLearned()
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, src_pos_embed, box_pos_embed, box_token_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        src_pos_embed = src_pos_embed.flatten(2).permute(2, 0, 1) # seq x bs x dim
        box_pos_embed = box_pos_embed.unsqueeze(2).repeat(1, 1, bs).transpose(1,2) # num_queries x bs x dim
        num_queries = box_pos_embed.shape[0]
        pos_embed = torch.cat([box_pos_embed, src_pos_embed])

        src = src.flatten(2).permute(2, 0, 1)
        box_token_embed = box_token_embed.unsqueeze(1).repeat(1, bs, 1)
        src = torch.cat([box_token_embed, src])

        output = src
        for i, layer in enumerate(self.encoder.layers):
            output, attn = layer(output, src_mask=None,
                                 src_key_padding_mask=None, pos=pos_embed, return_attn=True)
            n, s, _ = attn.shape
            for i in range(n):
                attn_maps_i = simple_nms(attn[i, :, :].unsqueeze(0)).squeeze(0)
                # Get top-k locations
                _, indices = torch.sort(attn_maps_i.flatten(), descending=True)
                indices = indices[:num_queries]
                x = indices % s
                y = indices // s
                pos_embed[:num_queries, i, :] = self.prior_pos_embed(x, y).squeeze(0)

        if self.encoder.norm is not None:
            output = self.encoder.norm(output)

        return output[:num_queries, :, :].transpose(0, 1)

def build_encoder_with_tokens(args):
    return TokenizedEncoder(d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


