from pathlib import Path

import scipy.io
import torch

from RTE_ATFPS_func import VDXY, basis_mask
from RTE_datagenerator import RTECoef
from RTE_dataloader import UnsuperviseDataset
from RTE_mgmodel import MG_precond, RTEUNetPrecond
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


def test_model_forward_shape_odd_grid_cpu():
    model = MG_precond(1, [4, 8], [4, 8], [1, 1])
    coeff = torch.randn(2, 3, 5, 5)
    rhs = torch.randn(2, 8, 5, 5)
    a_list, inv_a_list = model.setup(coeff)
    out = model(rhs, a_list, inv_a_list)
    assert out.shape == rhs.shape


def test_model_default_levels_batch_one_cpu():
    model = MG_precond(1)
    coeff = torch.randn(1, 3, 16, 16)
    rhs = torch.randn(1, 8, 16, 16)
    a_list, inv_a_list = model.setup(coeff)
    out = model(rhs, a_list, inv_a_list)
    assert out.shape == rhs.shape


def test_unet_preconditioner_shape_and_rhs_linearity_cpu():
    model = RTEUNetPrecond(1, channels=4, num_levels=2, init_output_scale=0.1)
    coeff = torch.randn(2, 3, 4, 4)
    rhs1 = torch.randn(2, 8, 4, 4)
    rhs2 = torch.randn(2, 8, 4, 4)
    alpha, beta = 1.7, -0.4
    a_list, inv_a_list = model.setup(coeff)
    out1 = model(rhs1, a_list, inv_a_list)
    out2 = model(rhs2, a_list, inv_a_list)
    out_mix = model(alpha * rhs1 + beta * rhs2, a_list, inv_a_list)
    assert out1.shape == rhs1.shape
    assert torch.linalg.norm(out1).item() == 0.0
    rel = torch.linalg.norm(out_mix - (alpha * out1 + beta * out2)) / torch.clamp(torch.linalg.norm(out_mix), min=1e-30)
    assert rel.item() < 1e-5


def test_unet_defect2_input_shape_and_zero_initialization_cpu():
    model = RTEUNetPrecond(1, channels=4, num_levels=2, input_channels=16, output_channels=8)
    coeff = torch.randn(2, 3, 4, 4)
    rhs = torch.randn(2, 16, 4, 4)
    a_list, inv_a_list = model.setup(coeff)
    out = model(rhs, a_list, inv_a_list)
    assert out.shape == (2, 8, 4, 4)
    assert torch.linalg.norm(out).item() == 0.0


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
