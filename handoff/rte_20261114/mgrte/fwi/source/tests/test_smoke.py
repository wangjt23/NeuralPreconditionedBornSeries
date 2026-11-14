from pathlib import Path

import scipy.io
import torch

from RTE_ATFPS_func import VDXY, basis_mask
from RTE_datagenerator import RTECoef
from RTE_dataloader import UnsuperviseDataset
from RTE_mgmodel import MG_precond
from gmres import fgmres
from rte_config import REPO_ROOT, dataset_filename, kernel_suffix


def test_dataset_filename_tokens():
    assert dataset_filename("diffusion", 1, 16, 1000, 5.0) == "diffusionN1I16d1000tol5.pt"
    assert kernel_suffix("0.2") == "g02"


def test_model_forward_shape_cpu():
    model = MG_precond(1, [4, 8], [4, 8], [1, 1])
    coeff = torch.randn(2, 3, 4, 4)
    rhs = torch.randn(2, 8, 4, 4)
    a_list, inv_a_list = model.setup(coeff)
    out = model(rhs, a_list, inv_a_list)
    assert out.shape == rhs.shape


def test_model_forward_shape_custom_channels_cpu():
    model = MG_precond(1, [4, 8], [4, 8], [1, 1], input_channels=4, output_channels=4)
    coeff = torch.randn(2, 3, 4, 4)
    rhs = torch.randn(2, 4, 4, 4)
    a_list, inv_a_list = model.setup(coeff)
    out = model(rhs, a_list, inv_a_list)
    assert out.shape == rhs.shape


def test_unsupervised_dataset_rhs_seed_is_reproducible():
    loaded = {
        "Coef": torch.zeros(2, 3, 4, 4),
        "fsmLRBTC": torch.zeros(2, 5, 4, 8, 4, 4),
        "I2A": torch.zeros(2, 8, 8, 4, 4),
        "MLRBT": torch.zeros(2, 4, 2, 4, 4, 4),
        "VecSize": torch.zeros(2, 4, 4, 4, dtype=torch.int32),
    }
    ds1 = UnsuperviseDataset(1, 4, 2, 3, loaded, seed_num=7)
    ds2 = UnsuperviseDataset(1, 4, 2, 3, loaded, seed_num=7)
    assert torch.equal(ds1.rhs, ds2.rhs)


def test_unsupervised_dataset_can_skip_fixed_rhs_storage():
    loaded = {
        "Coef": torch.zeros(2, 3, 4, 4),
        "fsmLRBTC": torch.zeros(2, 5, 4, 8, 4, 4),
        "I2A": torch.zeros(2, 8, 8, 4, 4),
        "MLRBT": torch.zeros(2, 4, 2, 4, 4, 4),
        "VecSize": torch.zeros(2, 4, 4, 4, dtype=torch.int32),
    }
    ds = UnsuperviseDataset(1, 4, 2, 3, loaded, seed_num=7, store_rhs=False)
    assert ds.rhs is None
    *_, rhs = ds[0]
    assert rhs.shape == (8, 4, 4)
    assert torch.count_nonzero(rhs).item() == 0


def test_unsupervised_dataset_can_use_replay_rhs_tensor():
    loaded = {
        "Coef": torch.zeros(2, 3, 4, 4),
        "fsmLRBTC": torch.zeros(2, 5, 4, 8, 4, 4),
        "I2A": torch.zeros(2, 8, 8, 4, 4),
        "MLRBT": torch.zeros(2, 4, 2, 4, 4, 4),
        "VecSize": torch.zeros(2, 4, 4, 4, dtype=torch.int32),
    }
    replay_rhs = torch.arange(3 * 8 * 4 * 4, dtype=torch.float32).reshape(3, 8, 4, 4)
    ds = UnsuperviseDataset(1, 4, 2, 5, loaded, seed_num=7, store_rhs=False, replay_rhs=replay_rhs)
    assert ds.rhs is not None
    *_, rhs0 = ds[0]
    *_, rhs4 = ds[4]
    assert torch.equal(rhs0, replay_rhs[0])
    assert torch.equal(rhs4, replay_rhs[1])


def test_all_regime_generates_valid_log_epsilon():
    data = RTECoef(2, 4, data_type="all", device="cpu")
    epsilon = torch.exp(data.coef[:, 2])
    assert torch.isfinite(epsilon).all()
    assert torch.all(epsilon > 0)
    assert torch.all(epsilon <= 1)


def test_basis_mask_device_and_shape():
    vec_size = torch.tensor([[[[1]], [[2]], [[0]], [[1]]]], dtype=torch.int32)
    mask = basis_mask(vec_size, M=1)
    assert mask.shape == (1, 8, 1, 1)
    assert mask.sum().item() == 4


def test_vdxy_y_basis_solves_y_eigenproblem():
    q = scipy.io.loadmat(REPO_ROOT / "discretized_parameters" / "quadrature2DN1.mat")
    kappa = scipy.io.loadmat(REPO_ROOT / "discretized_parameters" / "KappaN1g0.mat")["Kappa"]
    ct = torch.tensor(q["ct"]).squeeze(-1).to(torch.float32)
    st = torch.tensor(q["st"]).squeeze(-1).to(torch.float32)
    omega = torch.tensor(q["omega"]).squeeze(-1).to(torch.float32)
    M = int(q["M"].item())
    sigma_t = torch.full((1, 1), 2.0)
    sigma_a = torch.full((1, 1), 0.5)
    eps = torch.ones(1, 1)
    Kappa = torch.tensor(kappa).to(torch.float32)

    _, _, VY, DY = VDXY(sigma_t, sigma_a, eps, ct, st, omega, Kappa, M)
    ratio = 1 - eps[0, 0] ** 2 * sigma_a[0, 0] / sigma_t[0, 0]
    block_y = torch.diag(1 / st) @ (ratio * Kappa @ torch.diag(omega) - torch.eye(4 * M))
    for col in range(4 * M):
        vec = VY[:, col, 0, 0]
        lam = DY[col, 0, 0]
        rel = torch.linalg.norm(block_y @ vec - lam * vec) / torch.clamp(torch.linalg.norm(vec), min=1e-30)
        assert rel.item() < 1e-4


def test_fgmres_exact_right_preconditioner():
    diagonal = torch.linspace(1.0, 3.0, 8).reshape(1, 1, 2, 4)
    rhs = torch.randn(1, 1, 2, 4)
    A = lambda x: diagonal * x
    M = lambda r: r / diagonal
    _, history = fgmres(A, rhs, M, tol=1e-8, maxiter=2, verbose=False)
    assert history[-1] < 1e-6
