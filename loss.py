import torch.nn as nn
import torch

HINGE_LOSS_ENABLE = True
HINGE_LAMBDA = 0.2
DELTA = 2

class loss_func(nn.Module):
    def __init__(self, device):
        super(loss_func, self).__init__()
        self.device = device
        self.cross_entropy_layer = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        softmax_loss = self.cross_entropy_layer(logits, targets)
        age_prob = logits

        if HINGE_LOSS_ENABLE:
          sign_martix = torch.ones([age_prob.shape[0], age_prob.shape[1]])
          sign_martix = sign_martix.to(device=self.device)
          range_martix = torch.arange(0, 101).expand([age_prob.shape[0], 101]) # bn 101
          range_martix = range_martix.to(device=self.device)
          targets_expand = targets.expand([101, age_prob.shape[0]]).t()
          sign_martix[range_martix>=targets_expand] = -1 #when i >= target[n], sign_martix[n][i]=-1

          age_prob_part = age_prob[:, 1:]
          age_prob_move = torch.cat([age_prob_part, (age_prob[:, -1]).view([age_prob.shape[0], 1])], 1) #move to left by one column
          hinge = (age_prob-age_prob_move)*sign_martix #age 0 - age 1, age 1 - age 2...age target+2 - age target+1, ..., age 100 - age 99, age 100 - age 100

          hinge = hinge+DELTA
          zero_data = torch.zeros(hinge.shape).to(device=self.device)
          hinge = torch.max(torch.cat([hinge.unsqueeze(0), zero_data.unsqueeze(0)],0),0)[0]

          hinge_loss = (torch.sum(hinge,1)).mean()

        result = (softmax_loss + hinge_loss*HINGE_LAMBDA, softmax_loss, hinge_loss*HINGE_LAMBDA) if HINGE_LOSS_ENABLE else (softmax_loss,0,0)
        return result
