"""Validation-tunable repetition of a frozen learned correction block."""

import torch
from torch import nn


class RepeatedTransportCorrection(nn.Module):
    def __init__(self, rte, model, repeats=2):
        super().__init__()
        if repeats < 1:
            raise ValueError('positive repetition count required')
        self.rte, self.model, self.repeats = rte, model, repeats
        self.depth = model.depth*repeats
        self.factor_stages_per_apply = self.depth

    @torch.no_grad()
    def prepare(self):
        self.eval()
        self.model.prepare()
        return self

    def forward(self, residual):
        z, defect = torch.zeros_like(residual), residual
        for index in range(self.repeats):
            z = z+self.model(defect)
            if index+1 < self.repeats:
                defect = residual-self.rte.A_op(z)
        return z
