"""Tolerance-aware smooth log-excess on the ORIGINAL right-hand side.

No claim of consistency for total stopping time: this objective observes k=0..2.
The original-prefix factor must be measured BEFORE residual-RMS normalization.
"""
import math
import torch
import torch.nn.functional as F

TAU=1e-5
BETA=math.log(4.)
FLOOR=1e-12

def original_prefix_factor(raw_residual,original_rhs_norm):
    eta=raw_residual.flatten(1).norm(dim=1)/original_rhs_norm
    if not bool(torch.isfinite(eta).all()) or not bool((eta>0).all()):
        raise ValueError('finite positive original-RHS prefix factors required')
    return eta

def components(ratios,eta,kind):
    if ratios.ndim!=2 or eta.shape!=(len(ratios),):raise ValueError('batch/trajectory shape mismatch')
    logratios=ratios.clamp_min(FLOOR).log()
    excess=logratios+2*eta.log()[:,None]-2*math.log(TAU)
    if kind=='two_step':loss=logratios.mean()
    elif kind=='tolerance':loss=(BETA*F.softplus(excess/BETA)).mean()
    else:raise ValueError(kind)
    return loss,excess

def gradient_vector(grads,parameters):
    pieces=[]
    for g,p in zip(grads,parameters):
        g=torch.zeros_like(p) if g is None else g
        pieces.append((torch.view_as_real(g) if g.is_complex() else g).flatten().double())
    return torch.cat(pieces)

def gradient_comparison(old,new):
    if not bool(torch.isfinite(old).all()) or not bool(torch.isfinite(new).all()):
        raise ValueError('nonfinite diagnostic gradient')
    a=float(old.norm());b=float(new.norm())
    if not math.isfinite(a) or not math.isfinite(b) or min(a,b)<=0:raise ValueError('invalid diagnostic gradient norm')
    cosine=float(torch.dot(old,new)/(a*b))
    if not math.isfinite(cosine):raise ValueError('nonfinite diagnostic cosine')
    cosine=min(1.,max(-1.,cosine))
    scale=max(0.,float(torch.dot(old,new)/old.square().sum()))
    clipped_old=old/max(1.,a);clipped_new=new/max(1.,b)
    result=dict(old_norm=a,new_norm=b,cosine=cosine,one_minus_cosine=1-cosine,
        relative_difference=float((new-old).norm()/a),best_positive_scale=scale,
        relative_nonscaling_component=float((new-scale*old).norm()/a),
        clipped_relative_difference=float((clipped_new-clipped_old).norm()/clipped_old.norm()),
        real_component_sign_difference=float((old.sign()!=new.sign()).double().mean()))
    if not all(math.isfinite(x) for x in result.values()):raise ValueError('nonfinite gradient comparison')
    return result
