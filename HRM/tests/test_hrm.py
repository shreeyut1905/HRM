import pytest 

param = pytest.mark.parametrize
import torch

@param('casual',(False,False))
def test_hrm(casual):
    from HRM.hrm import HRM
    from x_transformers import Encoder 

    hrm = HRM(
        networks = [
            dict(
                dim=32,
                depth=2,
                attn_dim_head=8,
                heads = 1,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            ),
            dict(
                dim=32,
                depth=4,
                attn_dim_head=8,
                heads = 1,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            ),
            Encoder(
                dim=32,
                depth=0,
                attn_dim_head = 8,
                heads = 1,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False 
            )
        ],
        casual = casual,
        num_tokens = 256,
        dim= 32,
        reasoning_steps = 10 
    )
    seq = torch.randint(0,256,(3,1024))
    labels = torch.randint(0,256,(3,1024))

    loss , hiddens , _ = hrms(seq,labels = labels)
    loss.backward()

    loss,hiddens,_ =hrms(seq,hiddens=hiddens,labels=labels)
    loss.backward()

    pred = hrm(seq,reasoning_steps=10)
# @param('compute_loss_across_reasoning_steps',(False,True))
# @param('casual',(False,True))

if __name__ == '__main__':  
    test_hrm(casual=False)
