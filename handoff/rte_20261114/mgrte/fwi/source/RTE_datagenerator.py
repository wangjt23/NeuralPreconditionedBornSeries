from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import scipy.io
import os
import argparse
import time

try:
    from .rte_config import dataset_filename, kernel_suffix, load_yaml_config, merge_cli_defaults, resolve_device, set_seed
    from .utils import greenprint
    from .RTE_ATFPS_func import VDXY, FSM, Inflow2alpha, MLRO_Dirichlet, MBTO_Dirichlet
except ImportError:
    from rte_config import dataset_filename, kernel_suffix, load_yaml_config, merge_cli_defaults, resolve_device, set_seed
    from utils import greenprint
    from RTE_ATFPS_func import VDXY, FSM, Inflow2alpha, MLRO_Dirichlet, MBTO_Dirichlet

# 2d dense matrix
def read_bm_data(fn):
    # read by lines
    data = []
    with open(fn, 'r') as f:
        for line in f:
            data.append([float(i) for i in line.split()])
    return torch.tensor(data)

def fun_value(I0, J0, xl, xr, yl, yr, coeff, device):
    hx, hy = (xr - xl) / I0, (yr - yl) / J0
    x = torch.linspace(xl + 0.5 * hx, xr - 0.5 * hx, I0).to(device)
    y = torch.linspace(yl + 0.5 * hy, yr - 0.5 * hy, J0).to(device)
    xmesh, ymesh = torch.meshgrid(x, y, indexing='ij')

    num, _, n = coeff.shape
    value = torch.zeros([num, I0, J0]).to(device)

    for ins in range(num):
        value_x = torch.zeros([I0, J0]).to(device)
        value_y = torch.zeros([I0, J0]).to(device)
        coeff_ins = coeff[ins, :, :]
        for s in range(n):
            value_x += coeff_ins[0, s] * (xmesh - (xl + xr) / 2) ** s
            value_y += coeff_ins[1, s] * (ymesh - (yl + yr) / 2) ** s
        value[ins, :, :] = value_x * value_y

    return value

class RTECoef(Dataset):
    def __init__(self, size, num_coef, data_type='transport', device = 'cpu', xl=0, xr=1, yl=0, yr=1):
        self.num_coef = num_coef
        self.data_type = data_type
        self.size = size
        self.device = device
        self.xl = xl
        self.xr = xr
        self.yl = yl
        self.yr = yr
        [Sigma_a, Sigma_T] = self.get_sigma(num_coef)
        Varepsilon = self.get_epsilon(num_coef)
        Varepsilon_log = torch.log(Varepsilon)
        self.coef = torch.hstack((Sigma_T.reshape([num_coef,1,size,size]), Sigma_a.reshape([num_coef,1,size,size]), Varepsilon_log.reshape([num_coef,1,size,size])))
        print('data type:', data_type)
        
    def get_sigma(self, num):
        Sigma_a = torch.zeros(num,self.size,self.size).to(self.device)
        Sigma_T = torch.zeros(num,self.size,self.size).to(self.device)
        count = 0
        for n in range(1,5):
            sub_num = int(num / 4) if n < 4 else num - 3 * int(num / 4)
            sigma_T_pre = torch.rand((sub_num, 2, n)).to(self.device)
            sigma_T_pre[:, :, 0] = sigma_T_pre[:, :, 0] + 1
            sigma_T_pre[:, :, 1:n + 1] = 2 * sigma_T_pre[:, :, 1:n + 1] - 1
            Sigma_T[count:count + sub_num] = fun_value(self.size, self.size, self.xl, self.xr, self.yl, self.yr, sigma_T_pre, self.device)
            sigma_a_pre = torch.rand((sub_num, 2, n)).to(self.device)
            sigma_a_pre[:, :, 0] = sigma_a_pre[:, :, 0] + 1
            sigma_a_pre[:, :, 1:n + 1] = 2 * sigma_a_pre[:, :, 1:n + 1] - 1
            Sigma_a[count:count + sub_num] = fun_value(self.size, self.size, self.xl, self.xr, self.yl, self.yr, sigma_a_pre, self.device)
            count = count + sub_num
        indices = torch.randperm(num)
        shuffled_Sigma_T = Sigma_T[indices]
        indices = torch.randperm(num)
        shuffled_Sigma_a = Sigma_a[indices]
        shuffled_dvalue = torch.minimum((shuffled_Sigma_T - shuffled_Sigma_a - 0.2).amin(dim=(1, 2), keepdim=True), torch.zeros(1, device=self.device, dtype=Sigma_T.dtype).to(self.device))
        shuffled_Sigma_T = shuffled_Sigma_T - shuffled_dvalue
        return shuffled_Sigma_a, shuffled_Sigma_T
        
    
    def get_epsilon(self, num, data_type=None):
        data_type = data_type or self.data_type
        if data_type == 'transport':
            Varepsilon = torch.ones([num,self.size,self.size]).to(self.device)
        elif data_type == 'diffusion':
            Varepsilon = 0.01 * (torch.rand((num, 1, 1)).to(self.device) + 0.1) * torch.ones((1, self.size, self.size)).to(self.device)
        elif data_type == 'interface':
            Varepsilon = torch.ones([num,self.size,self.size]).to(self.device)
            trial_diffusion = 0.01 * (torch.rand(num).to(self.device) + 0.1)
            sub_num = int(num / 6)
            Varepsilon[:sub_num, :int(self.size / 2), :] = trial_diffusion[:sub_num].reshape([-1,1,1]) * torch.ones((int(self.size / 2), self.size)).to(self.device)
            Varepsilon[sub_num:2 * sub_num, int(self.size / 2):, :] = trial_diffusion[sub_num:2 * sub_num].reshape([-1,1,1]) * torch.ones((int(self.size / 2), self.size)).to(self.device)
            Varepsilon[2 * sub_num:3 * sub_num, :, :int(self.size / 2)] = trial_diffusion[2 * sub_num:3 * sub_num].reshape([-1,1,1]) * torch.ones((self.size, int(self.size / 2))).to(self.device)
            Varepsilon[3 * sub_num:4 * sub_num, :, int(self.size / 2):] = trial_diffusion[3 * sub_num:4 * sub_num].reshape([-1,1,1]) * torch.ones((self.size, int(self.size / 2))).to(self.device)
            Varepsilon[4 * sub_num:5 * sub_num, int(self.size / 4):3 * int(self.size / 4), int(self.size / 4):3 * int(self.size / 4)] = trial_diffusion[4 * sub_num:5 * sub_num].reshape([-1,1,1]) * torch.ones((int(self.size / 2), int(self.size / 2))).to(self.device)
            Varepsilon[5 * sub_num:] = trial_diffusion[5 * sub_num:].reshape([-1,1,1]) * torch.ones((self.size, self.size)).to(self.device)
            Varepsilon[5 * sub_num:, int(self.size / 4):3 * int(self.size / 4), int(self.size / 4):3 * int(self.size / 4)] = torch.ones((int(self.size / 2), int(self.size / 2))).to(self.device)
        elif data_type == 'bufferzone':
            Varepsilon = torch.ones([num,self.size,self.size]).to(self.device)
            count = 0
            for n in range(2, 6):
                sub_num = int(num / 4) if n < 4 else num - 3 * int(num / 4)
                if sub_num == 0:
                    continue
                varepsilon_pre = torch.rand((sub_num, 2, n)).to(self.device)
                varepsilon_pre[:, :, 0] = varepsilon_pre[:, :, 0] + 5
                varepsilon_pre[:, :, 1:n + 1] = 10 * varepsilon_pre[:, :, 1:n + 1] - 5
                Varepsilon_pre = fun_value(self.size, self.size, 0, 1, 0, 1, varepsilon_pre, self.device)
                # 获取每个切片的最小值和最大值
                min_vals = Varepsilon_pre.view(sub_num, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
                max_vals = Varepsilon_pre.view(sub_num, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)

                # 处理范围
                range_vals = max_vals - min_vals
                range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)

                # Min-Max归一化到[0, 1]
                normalized = (Varepsilon_pre - min_vals) / range_vals

                # 缩放到[0.001, 1]
                Varepsilon[count:count + sub_num] = normalized * 0.999 + 0.001
                count = count + sub_num
        elif data_type == 'all':
            Varepsilon = torch.ones([num,self.size,self.size]).to(self.device)
            sub_num = int(num / 4)
            Varepsilon[:sub_num] = self.get_epsilon(sub_num, 'transport')
            Varepsilon[sub_num:2 * sub_num] = self.get_epsilon(sub_num, 'diffusion')
            Varepsilon[2 * sub_num:3 * sub_num] = self.get_epsilon(sub_num, 'interface')
            Varepsilon[3 * sub_num:] = self.get_epsilon(num - 3 * sub_num, 'bufferzone')
        else:
            raise ValueError('data_type %s not supported' % data_type)
        return Varepsilon
    
    def getinfo(self, M, ct, st, omega, Kappa, tol):
        start_time = time.time()
        hx, hy = (self.xr-self.xl)/self.size, (self.yr-self.yl)/self.size
        fsmLRBTC = torch.zeros([self.num_coef, 5, 4*M, 8*M, self.size, self.size]).to(self.device)
        I2A = torch.zeros([self.num_coef, 8*M, 8*M, self.size, self.size]).to(self.device)
        MLRBT = torch.zeros([self.num_coef, 4, 2*M, 4*M, self.size, self.size]).to(self.device)
        VecSize = torch.zeros([self.num_coef, 4, self.size, self.size],dtype=int).to(self.device)
        for ins in range(self.num_coef):
            # T0 = time.time()
            [VX, DX, VY, DY] = VDXY(self.coef[ins,0], self.coef[ins,1], torch.exp(self.coef[ins,2]), ct, st, omega, Kappa, M, self.device)
            # T1 = time.time()
            [fsml_full, fsmr_full, fsmb_full, fsmt_full, fsmc_full] = FSM(VX, DX, VY, DY, self.xl, self.xr, self.yl, self.yr, self.coef[ins,0], torch.exp(self.coef[ins,2]), M, self.size, self.size, self.device)
            fsmLRBTC[ins, :, :, :, :, :] = torch.cat([fsml_full.unsqueeze(0), fsmr_full.unsqueeze(0), fsmb_full.unsqueeze(0), fsmt_full.unsqueeze(0), fsmc_full.unsqueeze(0)], dim=0)
            # T2 = time.time()
            I2A[ins, :, :, :, :] = Inflow2alpha(fsml_full, fsmr_full, fsmb_full, fsmt_full, M)
            # T3 = time.time()
            [Ml, Mr, VecSizel, VecSizer] = MLRO_Dirichlet(VX, DX, self.coef[ins,0], torch.exp(self.coef[ins,2]), hx, tol, M, self.size, self.size, self.device)
            [Mb, Mt, VecSizeb, VecSizet] = MBTO_Dirichlet(VY, DY, self.coef[ins,0], torch.exp(self.coef[ins,2]), hy, tol, M, self.size, self.size, self.device)
            MLRBT[ins, :, :, :, :] = torch.cat([Ml.unsqueeze(0), Mr.unsqueeze(0), Mb.unsqueeze(0), Mt.unsqueeze(0)], dim=0)
            VecSize[ins, :, :, :] = torch.cat([VecSizel.unsqueeze(0), VecSizer.unsqueeze(0), VecSizeb.unsqueeze(0), VecSizet.unsqueeze(0)], dim=0)
            # T4 = time.time()
            # print(f"T1-T0: {T1-T0} seconds, T2-T1: {T2-T1} seconds, T3-T2: {T3-T2} seconds, T4-T3: {T4-T3} seconds,")
        end_time = time.time()
        print(f'offline assembling completed using {end_time - start_time:.4f} seconds!')
        return self.coef, fsmLRBTC, I2A, MLRBT, VecSize
    
    
if __name__ == "__main__":
    # python RTE_datagenerator.py --data_type diffusion --num_coef 1000 --cuda_device cuda:0 --tol_exp 5;
    defaults = {
        'cuda_device': 'cuda:0',
        'data_type': 'diffusion',
        'num_coef': 1000,
        'mesh_size': 16,
        'N': 1,
        'tol_exp': 5,
        'kernel_g': '0',
        'seed': 1234,
    }
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', type=str, default=None)
    known, _ = bootstrap.parse_known_args()
    defaults = merge_cli_defaults(defaults, load_yaml_config(known.config))
    parser = argparse.ArgumentParser(description="RTE datagenerator with configurable parameters.")
    parser.add_argument('--config',          type = str,   nargs='?', default = None)
    parser.add_argument('--cuda_device',     type = str,   nargs='?', default = defaults['cuda_device'])
    parser.add_argument('--data_type',       type = str,   nargs='?', default = defaults['data_type'])
    parser.add_argument('--num_coef',        type = int,   nargs='?', default = defaults['num_coef'])
    parser.add_argument('--mesh_size',       type = int,   nargs='?', default = defaults['mesh_size'])
    parser.add_argument('--N',               type = int,   nargs='?', default = defaults['N'])
    parser.add_argument('--tol_exp',         type = float, nargs='?', default = defaults['tol_exp'])
    parser.add_argument('--kernel_g',        type = str,   nargs='?', default = defaults['kernel_g'])
    parser.add_argument('--seed',            type = int,   nargs='?', default = defaults['seed'])
    args = parser.parse_args()
    set_seed(args.seed)
    
    current_dir = os.path.dirname(__file__)
    quadrature_path = os.path.join(current_dir, 'discretized_parameters', 'quadrature2DN'+str(args.N)+'.mat')
    Kappa_path = os.path.join(current_dir, 'discretized_parameters', 'KappaN'+str(args.N)+kernel_suffix(args.kernel_g)+'.mat')
    coef_path = os.path.join(
        current_dir,
        'discretized_parameters',
        dataset_filename(args.data_type, args.N, args.mesh_size, args.num_coef, args.tol_exp),
    )
    
    device = resolve_device(args.cuda_device)
    greenprint(f"Using device: {device}")
    tol = args.tol_exp*np.log(10)
    quadrature = scipy.io.loadmat(quadrature_path)
    ct = torch.tensor(quadrature['ct']).squeeze(-1).to(torch.float32).to(device)
    st = torch.tensor(quadrature['st']).squeeze(-1).to(torch.float32).to(device)
    omega = torch.tensor(quadrature['omega']).squeeze(-1).to(torch.float32).to(device)
    M = quadrature['M'].item()
    Kappa = torch.tensor(scipy.io.loadmat(Kappa_path)['Kappa']).to(torch.float32).to(device)
    CoefData = RTECoef(args.mesh_size, args.num_coef, args.data_type, device)
    [Coef, fsmLRBTC, I2A, MLRBT, VecSize] = CoefData.getinfo(M, ct, st, omega, Kappa, tol)
    RTEData = {
        'Coef': Coef,
        'fsmLRBTC': fsmLRBTC,
        'I2A': I2A,
        'MLRBT': MLRBT,
        'VecSize': VecSize,
    }
    torch.save(RTEData, coef_path)
    loaded_data = torch.load(coef_path, weights_only=True)
    loaded_Coef = loaded_data['Coef']
    loaded_fsmLRBTC = loaded_data['fsmLRBTC']
    loaded_I2A = loaded_data['I2A']
    loaded_MLRBT = loaded_data['MLRBT']
    loaded_VecSize = loaded_data['VecSize']
    print(f"size of loaded Coef: {loaded_Coef.shape}")
    print(f"size of loaded fsmLRBTC: {loaded_fsmLRBTC.shape}")
    print(f"size of loaded I2A: {loaded_I2A.shape}")
    print(f"size of loaded MLRBT: {loaded_MLRBT.shape}")
    print(f"size of loaded VecSize: {loaded_VecSize.shape}")
    print(f"difference between original data and loaded data: {torch.linalg.norm(Coef-loaded_Coef)} and {torch.linalg.norm(fsmLRBTC-loaded_fsmLRBTC)} and {torch.linalg.norm(I2A-loaded_I2A)} and {torch.linalg.norm(MLRBT-loaded_MLRBT)} and {torch.linalg.norm((VecSize-loaded_VecSize).float())}")
    
