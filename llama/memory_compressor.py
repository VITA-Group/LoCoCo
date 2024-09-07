import os 
import sys 
import torch
import torch.nn as nn

from transformers.utils import logging

from einops import rearrange

from transformers.activations import ACT2FN
from utils.global_vars import get_args

logger = logging.get_logger(__name__)

def drop_tokens(past_key_value, layer_idx):

    args = get_args()

    logger.warning_once('Drop Token is Called.')
    
    past_key_states, past_value_states = past_key_value.key_cache[layer_idx], past_key_value.value_cache[layer_idx]
    hh_scores = past_key_value.hh_scores[layer_idx]

    bsz, num_heads, head_dim = past_key_states.shape[0], past_key_states.shape[1], past_key_states.shape[-1]

    zeros = torch.zeros_like(hh_scores, dtype=torch.bool, device=hh_scores.device)
    raw_length = past_key_states.shape[-2]

    if args.local_len > 0:
        _, hh_idxs = torch.topk(hh_scores[..., :-args.local_len], args.mem_size-args.local_len, dim=-1)
    else:
        _, hh_idxs = torch.topk(hh_scores, args.mem_size, dim=-1)
    
    mask_bottom = zeros.scatter(-1, hh_idxs, True)
    if args.local_len > 0:
        mask_bottom[:, :, -args.local_len:] = True
    
    mask_bottom = mask_bottom[..., None].repeat(1,1,1,head_dim)

    past_key_states = torch.masked_select(
        past_key_states, mask_bottom).reshape(bsz, num_heads, args.mem_size, head_dim)
    past_value_states = torch.masked_select(
        past_value_states, mask_bottom).reshape(bsz, num_heads, args.mem_size, head_dim)
    return past_key_states.detach().clone(), past_value_states.detach().clone()


class memory_saver(nn.Module):
    def __init__(self, config):
        super().__init__()

        args = get_args()

        self.config = config
        dim_kv = self.config.head_dim*2 
        self.mem_size = args.mem_size
        self.mem_compress = int(args.mem_size * (1-args.hh_keep_rate))
        self.keep_hh_size = self.mem_size - self.mem_compress
        self.local_len = args.local_len
        
        layers = []
        hidden_size, in_size = self.mem_compress * args.expand, dim_kv
        for i in range(args.n_convlayer):
            
            if i != 0:
                in_size = hidden_size
            if i == args.n_convlayer-1:
                hidden_size = self.mem_compress
            
            layers.append(nn.Conv1d(in_channels=in_size, out_channels=hidden_size, 
                            kernel_size=args.kernel_size, padding=int((args.kernel_size-1)//2)))
            
            if args.hidden_act is not None:
                layers.append(ACT2FN[args.hidden_act])
        
        self.layers = nn.Sequential(*layers)
        self.normalizer = torch.nn.Parameter(torch.ones(1)*args.normalizer_init)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data = torch.randn(m.weight.shape)
                # import pdb 
                # pdb.set_trace()
            # elif isinstance(m, nn.BatchNorm1d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()


    def make_residual_weight(self, non_heavy_hitter_hh_scores):

        # desirable shape [bsz, mem_compress, seqlen]
        bsz, n_head, seqlen = \
            non_heavy_hitter_hh_scores.shape[0], non_heavy_hitter_hh_scores.shape[1], non_heavy_hitter_hh_scores.shape[2]

        _, residual_idxs = torch.topk(non_heavy_hitter_hh_scores, self.mem_compress, dim=-1) # [bsz, mem_compress]
        residual_idxs = residual_idxs.sort(dim=-1, descending=False)[0][...,None]
        weight = torch.zeros([bsz, n_head, self.mem_compress, seqlen], 
                dtype=non_heavy_hitter_hh_scores.dtype, device=non_heavy_hitter_hh_scores.device)
        weight = weight.scatter(-1, residual_idxs, 1) 

        return weight

    
    def partition_past_key_values(self, past_key_states, past_value_states, hh_scores):

        bsz, n_head, seqlen = past_key_states.shape[0], past_key_states.shape[1], past_key_states.shape[2]

        zeros = torch.zeros_like(hh_scores, dtype=torch.bool, device=hh_scores.device)
        raw_length = past_key_states.shape[-2]

        if self.local_len > 0:
            _, hh_idxs = torch.topk(hh_scores[..., :-self.local_len], self.keep_hh_size-self.local_len, dim=-1)
        else:
            _, hh_idxs = torch.topk(hh_scores, self.keep_hh_size, dim=-1)
        
        mask_bottom = zeros.scatter(-1, hh_idxs, True)
        if self.local_len > 0:
            mask_bottom[:, :, -self.local_len:] = True
        
        mask_bottom = mask_bottom[..., None].repeat(1,1,1,self.config.head_dim)

        heavy_hitter_key_states = torch.masked_select(
            past_key_states, mask_bottom).reshape(bsz, n_head, self.keep_hh_size, self.config.head_dim)
        heavy_hitter_value_states = torch.masked_select(
            past_value_states, mask_bottom).reshape(bsz, n_head, self.keep_hh_size, self.config.head_dim)

        non_heavy_hitter_key_states = torch.masked_select(
            past_key_states, ~mask_bottom).reshape(bsz, n_head, seqlen-self.keep_hh_size, self.config.head_dim)
        non_heavy_hitter_value_states = torch.masked_select(
            past_value_states, ~mask_bottom).reshape(bsz, n_head, seqlen-self.keep_hh_size, self.config.head_dim)

        return (heavy_hitter_key_states, heavy_hitter_value_states), \
            (non_heavy_hitter_key_states, non_heavy_hitter_value_states, torch.masked_select(hh_scores, ~mask_bottom[...,0]).reshape(bsz, n_head, -1))


    def forward(self, past_key_value, layer_idx):

        past_key_states, past_value_states = past_key_value.key_cache[layer_idx].detach().clone(), past_key_value.value_cache[layer_idx].detach().clone()
        hh_scores = past_key_value.hh_scores[layer_idx].detach().clone()

        bsz, n_head, seqlen = past_key_states.shape[0], past_key_states.shape[1], past_key_states.shape[2]

        (heavy_hitter_key_states, heavy_hitter_value_states), (non_heavy_hitter_key_states, non_heavy_hitter_value_states, non_heavy_hitter_hh_scores) = \
            self.partition_past_key_values(past_key_states, past_value_states, hh_scores)
        
        residual_weight = self.make_residual_weight(non_heavy_hitter_hh_scores) # [bsz, mem_compress, seqlen]

        x = torch.cat([non_heavy_hitter_key_states, non_heavy_hitter_value_states], dim=-1)
        x = rearrange(x, 'b h s d -> (b h) d s', b=bsz, h=n_head)
        x = self.layers(x)
        x = nn.functional.softmax(x, dim=-1)
        weight = rearrange(x, '(b h) m s -> b h m s', b=bsz) 
        weight = residual_weight * (1-self.normalizer) + weight * self.normalizer
        # weight = residual_weight

        non_heavy_hitter_key_states = torch.matmul(weight, non_heavy_hitter_key_states)
        non_heavy_hitter_value_states = torch.matmul(weight, non_heavy_hitter_value_states)

        past_key_states = torch.cat([heavy_hitter_key_states, non_heavy_hitter_key_states], dim=-2)
        past_value_states = torch.cat([heavy_hitter_value_states, non_heavy_hitter_value_states], dim=-2)

        return past_key_states, past_value_states

