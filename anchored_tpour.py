"""Experimental fixed-teacher regularization for TPOUR; not a claim of a novel KD algorithm."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from tpour import TPOUR


def retention_loss(student, teacher, temperature=.1, representation_weight=1.):
    """KL over identical in-batch candidates + scale-relative embedding MSE.

    Teacher is detached. Cosine ranking reduces raw-score scale sensitivity;
    representation loss preserves directions AND norms for dot-product inference.
    Compute in FP32 outside autocast to keep softmax/KL numerically stable.
    """
    if temperature<=0 or representation_weight<0: raise ValueError('Invalid retention settings')
    device=student[0].device.type
    with torch.autocast(device_type=device,enabled=False):
        sq,sp,sr=[v.float() for v in student]
        tq,tp,tr=[v.detach().float() for v in teacher]
        sd=F.normalize(torch.cat([sp,sr]),dim=-1); td=F.normalize(torch.cat([tp,tr]),dim=-1)
        sl=F.normalize(sq,dim=-1)@sd.T/temperature
        tl=F.normalize(tq,dim=-1)@td.T/temperature
        kl=F.kl_div(F.log_softmax(sl,dim=-1),F.softmax(tl,dim=-1),reduction='batchmean')
        s=torch.cat([sq,sp,sr]); t=torch.cat([tq,tp,tr])
        representation=((s-t).square().sum(-1)/t.square().sum(-1).clamp_min(1e-6)).mean()
        total=kl+representation_weight*representation
    return total,{'anchor_kl':float(kl.detach()),'anchor_representation':float(representation.detach())}


class AnchoredTPOUR(TPOUR):
    def __init__(self,*args,anchor_weight=10.,anchor_temperature=.1,representation_weight=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if anchor_weight<0 or anchor_temperature<=0 or representation_weight<0:
            raise ValueError('Invalid anchor weight, temperature or representation weight')
        self.teacher=copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.anchor_weight=anchor_weight; self.anchor_temperature=anchor_temperature
        self.representation_weight=representation_weight

    def forward(self,queries,preferred,rejected):
        base,stats,keys,student=super().forward(queries,preferred,rejected,return_embeddings=True)
        self.teacher.eval()
        with torch.no_grad(): teacher=tuple(self.teacher(texts) for texts in (queries,preferred,rejected))
        anchor,extra=retention_loss(student,teacher,self.anchor_temperature,self.representation_weight)
        total=base+self.anchor_weight*anchor
        stats.update(extra,base_loss=float(base.detach()),anchor_loss=float(anchor.detach()),loss=float(total.detach()))
        return total,stats,keys
