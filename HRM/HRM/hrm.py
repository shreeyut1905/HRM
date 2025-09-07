from __future__ import annotations
from contextlib import nullcontext

import torch 
import torch.nn.functional as F
from torch import Tensor , tensor , is_tensor , cat , stack
from torch.nn import Embedding , Linear , Sequential , Module , ModuleList

from einops import rearrange , repeat
from einops.layers.torch import Rearrange , Reduce

from x_transformers import encoder , decoder , RMSNorm


# helper functions
def exists(v):
    return v is not None
def default(v,d):
    return v if exists(v) else d
def last(arr):
    return arr[-1]  
def divisible_by(num,den):
    return (num % den) == 0


class CombineHiddens(Module):
    def __init__(self,dim,num_hiddens_to_contact):
        super().__init__()
        self.num_hiddens_to_concat = num_hiddens_to_contact
        self.norms = ModuleList([RMSNorm(dim) for _ in range(num_hiddens_to_contact)])
        self.to_combined  = Linear(dim * self.num_hiddens_to_concat , dim , bias = False)
    def forward(self,hiddens:list[Tensor],hierarchy_index):
        hiddens_to_concat = hiddens[hierarchy_index:]
        assert len(hiddens_to_concat) == self.num_hiddens_to_concat


        normed = tuple(norm(t) for norm,t in zip(self.norms,hiddens))
        concatted = cat(normed,dim=-1)
        return self.to_combined(concatted)


class HRM(Module):
    def __init__(self,networks:list[Module|dict],*,dim,num_tokens,reasoning_steps=2,relative_period:int|tuple[int,...]=2,casual=False,ignore_index=-1):
        super().__init__() 
        attn_layers_Klass = Encoder if not casual else Decoder 
        self.casual = casual
        #input 
        self.to_input_embed = Embedding(num_tokens,dim)
        self.networks = ModuleList()
        for network in networks:
            if isinstance(network,dict):
                network = attn_layers_klass(**network)
            self.networks.append(network)
        self.num_networks = len(self.networks)
        assert self.num_networks > 0 

        num_higher_networks = self.num_networks - 1

        if not isinstance(relative_period,tuple):
            relative_period = (relative_period,)*num_higher_networks
        if len(relative_period) == (self.num_netowrks - 1):
            relative_period = (1,*relative_period)
        
        assert len(relative_period) == self.num_netowrks and relative_period[0]==1

        self.evaluate_networks_at = tensor(relative_period).cumprod(dim=-1).tolist()

        self.reasoning_step = reasoning_steps
        self.lowest_steps_per_reasoning_step = last(self.evaluate_networks_at)

        self.hidden_combiners = ModuleList([CombineHiddens(dim,self.num_networks + 1 - network_index) for network_index in range(self.num_networks)])
        self.to_pred = Linear(dim,num_tokens,bias=False)

        self.ignore_index = ignore_index
    def forward(self,seq,hiddens:tuple[Tensor,...]|None = None,*,labels=None,detach_hiddens=True,one_step_grad=True,reasoning_steps=None,return_autoreg_loss=False):
        if return_autoreg_loss:
            assert self.casual and not exists(labels)
            seq,labels = seq[:,:-1],seq[:,1:]
        return_loss = exists(labels)

        reasoning_steps = default(reasoning_steps,self.reasoning_steps)

        if exists(hiddens) and detach_hiddens:
            hiddens = tuple(h.detach() for h in hiddens)
        tokens = self.to_input_embed(seq)

        if not exists(hiddens):
            hiddens = torch.zero_like(tokens)
            hiddens  = repeat(hiddens,'... -> num_networks ...',num_networks = self.num_networks)
        assert len(hiddens) == self.num_networks
        hiddens_dict = {index:hidden for index,hidden in enumerate(hiddens)}

        total_low_steps = reasoning_steps * self.lowest_steps_per_reasoning_step

        for index in range(self.total_low_steps):
            iteration = index + 1
            is_last_step = index == (total_low_steps - 1)

            context = torch.no_grad if one_step_grad and not is_last_step else nullcontext
            with context():
                for network_index ,(network,hidden_combine,evaluate_netowrk_at) in enumerate(zip(self.networks,self.hidden_combiners,self.evaluate_networks_at)):
                    if not divisible_by(iteration,evaluate_netowrk_at):
                        continue
                    all_hiddens = (
                        tokens,
                        *hiddens_dict.values()
                    )
                    combined_input = hidden_combine(all_hiddens,network_index)
                    next_hidden = network(combined_input)
                    hiddens_dict[network_index] = next_hidden
            
        highest_hidden = hiddens_dict[self.num_networks - 1]
        pred = self.to_pred(highest_hidden)
        hiddens_out = hiddens_dict.values()
        if not return_loss:
            return pred,hiddens_out
        loss = F.cross_entropy(
            rearrange(pred,'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index
        )
        return loss , hiddens_out , pred
        

    


