from types import SimpleNamespace

import torch

from RTE_ATFPS_func import func_inflow_Dirichlet
from rte_preconditioning import mgnet_preconditioner
from train import residual_loss


class IdentityModel(torch.nn.Module):
    def setup(self, coeff):
        return None, None

    def forward(self, rhs, a_list, inv_a_list):
        return rhs


class ChannelScaleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, 8, 1, 1))

    def setup(self, coeff):
        return None, None

    def forward(self, rhs, a_list, inv_a_list):
        return self.weight * rhs


def make_diagonal_atfps_batch(scale, rhs):
    batch_size = rhs.shape[0]
    M = 1
    mesh_size = 1
    scale = scale.to(rhs.dtype)
    coeff = torch.zeros(batch_size, 3, mesh_size, mesh_size)
    I2A = torch.eye(8).reshape(1, 8, 8, 1, 1).repeat(batch_size, 1, 1, 1, 1)
    fsmLRBTC = torch.zeros(batch_size, 5, 4, 8, mesh_size, mesh_size)
    MLRBT = torch.zeros(batch_size, 4, 2, 4, mesh_size, mesh_size)
    vec_size = torch.full((batch_size, 4, mesh_size, mesh_size), 2, dtype=torch.int32)

    boundary_rows = ([3, 0], [1, 2], [0, 1], [2, 3])
    diagonal_rows = ([2, 1], [0, 3], [2, 3], [0, 1])
    for face, base_channel in enumerate([0, 2, 4, 6]):
        for local in range(2):
            channel = base_channel + local
            fsmLRBTC[:, face, boundary_rows[face][local], channel, 0, 0] = 1.0
            fsmLRBTC[:, face, diagonal_rows[face][local], channel, 0, 0] = scale[channel]
            MLRBT[:, face, local, diagonal_rows[face][local], 0, 0] = 1.0

    return coeff, fsmLRBTC, I2A, MLRBT, vec_size, rhs, M


def test_born_coordinate_loss_beats_raw_for_block_diagonal_reference():
    rhs = torch.linspace(-1.0, 1.0, 16).reshape(2, 8, 1, 1)
    scale = torch.tensor([2.0, 3.0, 1.5, 2.5, 4.0, 1.25, 2.25, 3.5])
    batch = make_diagonal_atfps_batch(scale, rhs)
    M = batch[-1]
    model = IdentityModel()
    loss_fn = torch.nn.MSELoss()

    raw_args = SimpleNamespace(
        mesh_size=1,
        loss_type="raw",
        preconditioner_form="direct",
        physical_loss_weight=0.1,
        rhs_mode="fixed",
        loss_normalization="relative",
    )
    born_args = SimpleNamespace(
        mesh_size=1,
        loss_type="born_mixed",
        preconditioner_form="direct",
        physical_loss_weight=0.1,
        rhs_mode="fixed",
        loss_normalization="relative",
    )

    raw_loss = residual_loss(model, batch[:-1], M, raw_args, torch.device("cpu"), loss_fn)
    born_loss = residual_loss(model, batch[:-1], M, born_args, torch.device("cpu"), loss_fn)

    assert raw_loss.item() > 0.1
    assert born_loss.item() < 1e-6


def test_born_mgnet_preconditioner_matches_block_diagonal_operator():
    rhs = torch.randn(2, 8, 1, 1)
    scale = torch.tensor([2.0, 3.0, 1.5, 2.5, 4.0, 1.25, 2.25, 3.5])
    coeff, fsmLRBTC, I2A, MLRBT, vec_size, rhs, M = make_diagonal_atfps_batch(scale, rhs)
    model = IdentityModel()
    a_list, inv_a_list = model.setup(coeff)

    raw_candidate = mgnet_preconditioner(
        rhs, model, a_list, inv_a_list, I2A, fsmLRBTC, MLRBT, vec_size, M, 1, torch.device("cpu"), mode="raw"
    )
    born_candidate = mgnet_preconditioner(
        rhs, model, a_list, inv_a_list, I2A, fsmLRBTC, MLRBT, vec_size, M, 1, torch.device("cpu"), mode="born"
    )

    A_raw = func_inflow_Dirichlet(raw_candidate, I2A, fsmLRBTC, MLRBT, vec_size, M, 1, 1)
    A_born = func_inflow_Dirichlet(born_candidate, I2A, fsmLRBTC, MLRBT, vec_size, M, 1, 1)
    raw_rel = torch.linalg.norm(A_raw - rhs) / torch.linalg.norm(rhs)
    born_rel = torch.linalg.norm(A_born - rhs) / torch.linalg.norm(rhs)

    assert raw_rel.item() > 0.1
    assert born_rel.item() < 1e-6


def test_born_training_loss_has_effective_gradient():
    rhs = torch.randn(4, 8, 1, 1)
    scale = torch.tensor([2.0, 3.0, 1.5, 2.5, 4.0, 1.25, 2.25, 3.5])
    batch = make_diagonal_atfps_batch(scale, rhs)
    M = batch[-1]
    model = ChannelScaleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.2)
    loss_fn = torch.nn.MSELoss()
    args = SimpleNamespace(
        mesh_size=1,
        loss_type="born_mixed",
        preconditioner_form="direct",
        physical_loss_weight=0.1,
        rhs_mode="fixed",
        loss_normalization="relative",
    )

    losses = []
    for _ in range(30):
        loss = residual_loss(model, batch[:-1], M, args, torch.device("cpu"), loss_fn)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < 0.25 * losses[0]
