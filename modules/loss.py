import torch.nn as nn
import torch
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):

        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0

class KL(nn.Module):
    def __init__(self, ):
        super(KL, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        kl = F.kl_div(logpt0, logpt1, reduction='mean')
        return kl

class LossFactory:
    @staticmethod
    def get_loss(config_loss):
        if config_loss == 'clip':
            return CLIPLoss()
        else:
            raise NotImplemented
