from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import argparse

try:
    from .rte_config import dataset_filename, resolve_device
    from .utils import greenprint
except ImportError:
    from rte_config import dataset_filename, resolve_device
    from utils import greenprint

def fun_value(I0, J0, xl, xr, yl, yr, coeff):
    hx, hy = (xr - xl) / I0, (yr - yl) / J0
    x = torch.linspace(xl + 0.5 * hx, xr - 0.5 * hx, I0)
    y = torch.linspace(yl + 0.5 * hy, yr - 0.5 * hy, J0)
    xmesh, ymesh = torch.meshgrid(x, y, indexing='ij')  # 添加 indexing 参数以避免警告

    num, _, n = coeff.shape
    value = torch.zeros([num, I0, J0])

    for ins in range(num):
        value_x = torch.zeros([I0, J0])
        value_y = torch.zeros([I0, J0])
        coeff_ins = coeff[ins, :, :]
        for s in range(n):
            value_x += coeff_ins[0, s] * (xmesh - (xl + xr) / 2) ** s
            value_y += coeff_ins[1, s] * (ymesh - (yl + yr) / 2) ** s
        value[ins, :, :] = value_x * value_y

    return value

class UnsuperviseDataset(Dataset):
    def __init__(self, N, size, num_coef, num_data, loaded_data, device='cpu', seed_num = 33, store_rhs=True):
        # indices = torch.randperm(10000)[:num_coef]
        # self.Coef = loaded_data['Coef'][indices]
        # self.fsmLRBTC = loaded_data['fsmLRBTC'][indices]
        # self.I2A = loaded_data['I2A'][indices]
        # self.MLRBT = loaded_data['MLRBT'][indices]
        # self.VecSize = loaded_data['VecSize'][indices]
        self.Coef = loaded_data['Coef']
        self.fsmLRBTC = loaded_data['fsmLRBTC']
        self.I2A = loaded_data['I2A']
        self.MLRBT = loaded_data['MLRBT']
        self.VecSize = loaded_data['VecSize']
        # M=N(N+1)/2
        self.rhs_shape = (4*N*(N+1), size, size)
        self.store_rhs = store_rhs
        if store_rhs:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(seed_num))
            self.rhs = (2*torch.rand((num_data, *self.rhs_shape), generator=generator)-1).to(device)
        else:
            self.rhs = None
        # self.rhs = torch.rand((num_data, 4*N*(N+1), size, size)).to(device)
        self.num_coef = num_coef
        self.num_data = num_data
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        # idx should between 0 and self.num_coef*self.num_rhs-1
        rhs = self.rhs[idx] if self.rhs is not None else torch.zeros(self.rhs_shape)
        return self.Coef[idx % self.num_coef], self.fsmLRBTC[idx % self.num_coef], self.I2A[idx % self.num_coef], self.MLRBT[idx % self.num_coef], self.VecSize[idx % self.num_coef], rhs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTE dataloader with configurable parameters.")
    parser.add_argument('--cuda_device',     type = str,   nargs='?', default = 'cuda:0')
    parser.add_argument('--data_type',       type = str,   nargs='?', default = 'transport')
    parser.add_argument('--num_coef',        type = int,   nargs='?', default = 2000)
    parser.add_argument('--num_data',        type = int,   nargs='?', default = 20000)
    parser.add_argument('--mesh_size',       type = int,   nargs='?', default = 16)
    parser.add_argument('--N',               type = int,   nargs='?', default = 1)
    parser.add_argument('--n',               type = int,   nargs='?', default = 2)
    parser.add_argument('--tol_exp',         type = float, nargs='?', default = 5)
    parser.add_argument('--seed',            type = int,   nargs='?', default = 33)
    args = parser.parse_args()
    
    current_dir = os.path.dirname(__file__)
    coef_path = os.path.join(
        current_dir,
        'discretized_parameters',
        dataset_filename(args.data_type, args.N, args.mesh_size, args.num_coef, args.tol_exp),
    )

    device = resolve_device(args.cuda_device)
    greenprint(f"Using device: {device}")
    tol = args.tol_exp*np.log(10)
    loaded_data = torch.load(coef_path, map_location='cpu', weights_only=True)
    M = int(args.N*(args.N+1)/2)
    dataset = UnsuperviseDataset(args.N, args.mesh_size, args.num_coef, args.num_data, loaded_data, device, args.seed)
    dataloader = DataLoader(dataset, batch_size=500, shuffle=False)
    for i, [Coef_ins, fsmLRBTC_ins, I2A_ins, MLRBT_ins, VecSize_ins, rhs_ins] in enumerate(dataloader):
        print(Coef_ins.shape, fsmLRBTC_ins.shape, I2A_ins.shape, MLRBT_ins.shape, VecSize_ins.shape, rhs_ins.shape)
