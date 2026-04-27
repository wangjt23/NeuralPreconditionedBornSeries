import os
import torch
import random
import scipy.io
from scipy.ndimage import zoom
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataclasses import dataclass
from utils import *
## dataloader
class unspvd_sos_Dataset(Dataset):
    def __init__(self, sos_path, data_size=480, data_type='openbreastus', data_num=2000, test=False, test_point=6000):
        self.data_type = data_type
        super().__init__()
        if data_type=='openbreastus':
            self.sos_data = []
            for i in range(data_num):
                if not test:
                    index_i = i+1
                else:
                    index_i = i+test_point
                self.sos_data.append(np.load(os.path.join(sos_path, f'train_{index_i}.npy')))
            self.sos_data = torch.from_numpy(np.array(self.sos_data)).unsqueeze(1) # [N, 1, 480, 480]
        elif data_type=='openbreastus_inter':
            self.sos_data = []
            for i in range(data_num):
                if not test:
                    index_i = i+1
                else:
                    index_i = i+test_point
                self.sos_data.append(np.load(os.path.join(sos_path, f'train_{index_i}.npy')))
            self.sos_data = torch.from_numpy(np.array(self.sos_data)).unsqueeze(1) # [N, 1, 480, 480]
            self.sos_data = F.interpolate(self.sos_data, size=(data_size, data_size), mode='bilinear', align_corners=False)
        elif data_type=='openbreastus_pad':
            self.sos_data = []
            for i in range(data_num):
                if not test:
                    index_i = i+1
                else:
                    index_i = i+test_point
                self.sos_data.append(np.load(os.path.join(sos_path, f'train_{index_i}.npy')))
            self.sos_data = torch.from_numpy(np.array(self.sos_data)).unsqueeze(1) # [N, 1, 480, 480]
            padding_size = (data_size - self.sos_data.size(-1))//2
            self.sos_data = F.pad(self.sos_data, pad=(padding_size, padding_size, padding_size, padding_size), mode='replicate')
        elif data_type in ("openfwi_CurveVel_A", "openfwi_CurveVel_B", "openfwi_FlatVel_A"):
            if not test:
                # faster
                num_file = int(np.ceil(data_num / 500))
                self.sos_data = []
                for i in range(num_file):
                    index_i = i+1
                    self.sos_data.append(np.load(os.path.join(sos_path, f"model{index_i}.npy")))
                self.sos_data = torch.from_numpy(np.concatenate(self.sos_data,axis=0))
                self.sos_data = self.sos_data[:data_num] # [N, 1, 70, 70]
            else:
                # slower
                self.sos_data = []
                for i in range(data_num):
                    idx = test_point + i
                    file_id = idx // 500 + 1
                    data_id = idx % 500
                    self.sos_data.append(np.load(os.path.join(sos_path, f"model{file_id}.npy"))[data_id])
                
                self.sos_data = torch.from_numpy(np.array(self.sos_data)) # [N, 1, 70, 70]
            self.sos_data = F.interpolate(self.sos_data, size=(256, 256), mode='bilinear', align_corners=False)
        
        elif data_type in ("openfwi_CurveVel_A_pad", "openfwi_CurveVel_B_pad", "openfwi_FlatVel_A_pad"):
            if not test:
                # faster
                num_file = int(np.ceil(data_num / 500))
                self.sos_data = []
                for i in range(num_file):
                    index_i = i+1
                    self.sos_data.append(np.load(os.path.join(sos_path, f"model{index_i}.npy")))
                self.sos_data = torch.from_numpy(np.concatenate(self.sos_data,axis=0))
                self.sos_data = self.sos_data[:data_num] # [N, 1, 70, 70]
            else:
                # slower
                self.sos_data = []
                for i in range(data_num):
                    idx = test_point + i
                    file_id = idx // 500 + 1
                    data_id = idx % 500
                    self.sos_data.append(np.load(os.path.join(sos_path, f"model{file_id}.npy"))[data_id])
                
                self.sos_data = torch.from_numpy(np.array(self.sos_data)) # [N, 1, 70, 70]
            self.sos_data = F.interpolate(self.sos_data, size=(data_size, data_size), mode='bilinear', align_corners=False)
            # self.sos_data = F.interpolate(self.sos_data, size=(256, 256), mode='bilinear', align_corners=False)
            # padding_size = (data_size - self.sos_data.size(-1))//2
            # self.sos_data = F.pad(self.sos_data, pad=(padding_size, padding_size, padding_size, padding_size), mode='replicate')
        elif data_type == "openfwi_CurveFault_B":
            if not test:
                # faster
                num_file  = int(np.ceil(data_num / 500))
                self.sos_data = []
                for i in range(num_file):
                    index_i = i
                    self.sos_data.append(np.load(os.path.join(sos_path, f"vel{(index_i//36)+6}_1_{index_i%36}.npy")))
                self.sos_data = torch.from_numpy(np.concatenate(self.sos_data,axis=0))
                self.sos_data = self.sos_data[:data_num] # [N, 1, 70, 70]
            else:
                # slower
                self.sos_data = []
                for i in range(data_num):
                    idx = test_point + i
                    file_id = idx // 500
                    data_id = idx % 500
                    self.sos_data.append(np.load(os.path.join(sos_path, f"vel{(file_id//36)+6}_1_{file_id%36}.npy"))[data_id])
                self.sos_data = torch.from_numpy(np.array(self.sos_data)) # [N, 1, 70, 70]
            self.sos_data = F.interpolate(self.sos_data, size=(data_size, data_size), mode='bilinear', align_corners=False)
        elif data_type == "openfwi_CurveFault_B_pad":
            if not test:
                # faster
                num_file  = int(np.ceil(data_num / 500))
                self.sos_data = []
                for i in range(num_file):
                    index_i = i
                    self.sos_data.append(np.load(os.path.join(sos_path, f"vel{(index_i//36)+6}_1_{index_i%36}.npy")))
                self.sos_data = torch.from_numpy(np.concatenate(self.sos_data,axis=0))
                self.sos_data = self.sos_data[:data_num] # [N, 1, 70, 70]
            else:
                # slower
                self.sos_data = []
                for i in range(data_num):
                    idx = test_point + i
                    file_id = idx // 500
                    data_id = idx % 500
                    self.sos_data.append(np.load(os.path.join(sos_path, f"vel{(file_id//36)+6}_1_{file_id%36}.npy"))[data_id])
                self.sos_data = torch.from_numpy(np.array(self.sos_data)) # [N, 1, 70, 70]
            self.sos_data = F.interpolate(self.sos_data, size=(256, 256), mode='bilinear', align_corners=False)
            padding_size = (data_size - self.sos_data.size(-1))//2
            self.sos_data = F.pad(self.sos_data, pad=(padding_size, padding_size, padding_size, padding_size), mode='replicate')
        elif data_type == "openfwi3d":
            self.sos_data = []
            index = 0
            pieces = [i for i in range(1,64)]
            if not test:
                years = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35]
            else:
                years = [100]

            for year in years:
                for piece in pieces:
                    if index < data_num:
                        with open (os.path.join(sos_path, f"year{year}_cut{piece}.bin")) as f:
                            model = np.fromfile(f,np.float32)
                            model=model.reshape(400,400,350)
                            f.close() 
                        self.sos_data.append(model)
                        index+=1
                    else:
                        break
            
            self.sos_data = torch.from_numpy(np.stack(self.sos_data, axis=0)).unsqueeze(1)
            self.sos_data = self.sos_data[:data_num] # [N, 400, 400, 350]

            self.sos_data = F.interpolate(self.sos_data, size=(data_size, data_size, data_size), mode='trilinear', align_corners=False)
        elif data_type == "marmousi":
            vp = read_marmousi_vp(sos_path)
            zoom_factors = (1024/751, 1024/2301)
            vp = zoom(vp, zoom_factors, order=3)
            if data_size==256:
                # faster
                H, W = vp.shape
                self.sos_data = []
                for i in range(data_num):
                    top = np.random.randint(0, H - data_size + 1)
                    left = np.random.randint(0, W - data_size + 1)
                    patch = vp[top:top+data_size, left:left+data_size]
                    self.sos_data.append(patch)
                self.sos_data = torch.from_numpy(np.stack(self.sos_data,axis=0))
                self.sos_data = self.sos_data.unsqueeze(1) # N, 1, 256, 256
            else:
               # for validation
                self.sos_data = torch.from_numpy(vp).unsqueeze(0).unsqueeze(0) # [N, 1, 1024, 1024]

        else:
            raise NotImplementedError(f"{data_type} dataset not implemented.")
        
        
    def __len__(self):
        return len(self.sos_data)
    
    def __getitem__(self, index):
        sos_i   = self.sos_data[index]
        return sos_i


## Data configurations
@dataclass
class Dataconfig_openbreastus:
    size:       int   = 480
    freq:       float = 1.2e5 # k0 = 500, ppw = 6
    dx:         float = 1/480 # normalized domain
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus'

@dataclass
class Dataconfig_openbreastus6:
    size:       int   = 480
    freq:       float = 1.2e5 # k0 = 500, ppw = 6
    dx:         float = 1/480 # normalized domain, (m)
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/home/wangjt/work/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus'

@dataclass
class Dataconfig_openbreastus8:
    size:       int   = 480
    freq:       float = 3/4*1.2e5 # k0 = 375, ppw = 8
    dx:         float = 1/480 # normalized domain, (m)
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/home/wangjt/work/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus'

@dataclass
class Dataconfig_openbreastus10:
    size:       int   = 480
    freq:       float = 3/5*1.2e5 # k0 = 300, ppw = 10
    dx:         float = 1/480 # normalized domain, (m)
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/home/wangjt/work/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus'

@dataclass
class Dataconfig_openbreastus12:
    size:       int   = 480
    freq:       float = 1/2*1.2e5 # k0 = 250, ppw = 12
    dx:         float = 1/480 # normalized domain, (m)
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/home/wangjt/work/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus'

@dataclass
class Dataconfig_openbreastus14:
    size:       int   = 480
    freq:       float = 3/7*1.2e5 # k0 = 214, ppw = 14
    dx:         float = 1/480 # normalized domain, (m)
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/home/wangjt/work/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus'

@dataclass
class Dataconfig_openbreastus16:
    size:       int   = 480
    freq:       float = 3/8*1.2e5 # k0 = 188, ppw = 16
    dx:         float = 1/480 # normalized domain, (m)
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/home/wangjt/work/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus'

@dataclass
class Dataconfig_openbreastus_inter:
    size:       int   = 4096
    freq:       float = 1.2e5 # ppw = 6
    dx:         float = 1/480 # since linear property, we use 1/480 for convenient
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/home/wangjt/work/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus_inter'

@dataclass
class Dataconfig_openbreastus_pad:
    size:       int   = 512 # change 
    freq:       float = 1.2e5 # ppw = 6
    dx:         float = 1/480 # normalized domain
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus_pad'

@dataclass
class Dataconfig_openbreastus_pad512:
    size:       int   = 512 # change 
    freq:       float = 1.2e5 # ppw = 6
    dx:         float = 1/480 # normalized domain
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus_pad'

@dataclass
class Dataconfig_openbreastus_pad1024:
    size:       int   = 1024 # change 
    freq:       float = 1.2e5 # ppw = 6
    dx:         float = 1/480 # normalized domain
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus_pad'

@dataclass
class Dataconfig_openbreastus_pad2048:
    size:       int   = 2048 # change 
    freq:       float = 1.2e5 # ppw = 6
    dx:         float = 1/480 # normalized domain
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus_pad'

@dataclass
class Dataconfig_openbreastus_pad4096:
    size:       int   = 4096 # change 
    freq:       float = 1.2e5 # ppw = 6
    dx:         float = 1/480 # normalized domain
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus_pad'

@dataclass
class Dataconfig_openbreastus_pad8192:
    size:       int   = 8192 # change 
    freq:       float = 1.2e5 # ppw = 6
    dx:         float = 1/480 # normalized domain
    data_num:   int   = 7200 
    sos_mean:   float = 1500.0
    sos_max:    float = 1609.7
    sos_min:    float = 1393.9
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openbreastus/speed'
    data_type:  str   = 'openbreastus_pad'

@dataclass
class Dataconfig_openfwi_CurveVel_B:
    size:       int   = 256
    freq:       float = 64000.0 # ppw = 12
    # freq:       float = 54857.1 # ppw=14
    # freq:       float = 48000. # ppw=16
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/home/wangjt/work/datasets/openfwi/CurveVel_B'
    data_type:  str   = 'openfwi_CurveVel_B'

@dataclass
class Dataconfig_openfwi_CurveFault_B:
    size:       int   = 256
    freq:       float = 64935.# ppw = 12
    dx:         float = 1./256
    data_num:   int   = 54000 
    sos_mean:   float = 3035.7
    sos_max:    float = 4499.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveFault_B'
    data_type:  str   = 'openfwi_CurveFault_B'


@dataclass
class Dataconfig_openfwi_CurveFault_B14:
    size:       int   = 256
    freq:       float = 54857.1# ppw = 14
    dx:         float = 1./256
    data_num:   int   = 54000 
    sos_mean:   float = 3035.7
    sos_max:    float = 4499.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveFault_B'
    data_type:  str   = 'openfwi_CurveFault_B'


@dataclass
class Dataconfig_openfwi_CurveFault_B16:
    size:       int   = 256
    freq:       float = 48000.# ppw = 16
    dx:         float = 1./256
    data_num:   int   = 54000 
    sos_mean:   float = 3035.7
    sos_max:    float = 4499.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveFault_B'
    data_type:  str   = 'openfwi_CurveFault_B'

@dataclass
class Dataconfig_openfwi_CurveFault_B18:
    size:       int   = 256
    freq:       float = 42666.66
    dx:         float = 1./256
    data_num:   int   = 54000 
    sos_mean:   float = 3035.7
    sos_max:    float = 4499.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveFault_B'
    data_type:  str   = 'openfwi_CurveFault_B'

@dataclass
class Dataconfig_openfwi_CurveFault_B20:
    size:       int   = 256
    freq:       float = 38400.
    dx:         float = 1./256
    data_num:   int   = 54000 
    sos_mean:   float = 3035.7
    sos_max:    float = 4499.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveFault_B'
    data_type:  str   = 'openfwi_CurveFault_B'

@dataclass
class Dataconfig_openfwi_CurveFault_B24:
    size:       int   = 256
    freq:       float = 32000.# ppw = 24
    dx:         float = 1./256
    data_num:   int   = 54000 
    sos_mean:   float = 3035.7
    sos_max:    float = 4499.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveFault_B'
    data_type:  str   = 'openfwi_CurveFault_B'


@dataclass
class Dataconfig_openfwi_CurveFault_B_pad:
    size:       int   = 256
    freq:       float = 64935. # ppw = 12
    dx:         float = 1./256
    data_num:   int   = 54000 
    sos_mean:   float = 3035.7
    sos_max:    float = 4499.
    sos_min:    float = 1500.
    dir_path:   str   = '/home/wangjt/work/datasets/openfwi/CurveFault_B'
    data_type:  str   = 'openfwi_CurveFault_B_pad'



@dataclass
class Dataconfig_openfwi_CurveVel_A:
    size:       int   = 256
    freq:       float = 64000.# ppw = 12
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A'

@dataclass
class Dataconfig_openfwi_CurveVel_A_pad512:
    size:       int   = 512
    freq:       float = 64000.# ppw = 12
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A_pad'



@dataclass
class Dataconfig_openfwi_CurveVel_A_pad128:
    size:       int   = 128
    freq:       float = 64000.# ppw = 12
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A_pad'


@dataclass
class Dataconfig_openfwi_CurveVel_A_pad2048:
    size:       int   = 2048
    freq:       float = 64000.# ppw = 12
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A_pad'


@dataclass
class Dataconfig_openfwi_CurveVel_A14:
    size:       int   = 256
    freq:       float = 54857.1# ppw = 14
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A'


@dataclass
class Dataconfig_openfwi_CurveVel_A16:
    size:       int   = 256
    freq:       float = 48000.# ppw = 16
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A'

@dataclass
class Dataconfig_openfwi_CurveVel_A20:
    size:       int   = 256
    freq:       float = 48000.# ppw = 16
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A'

@dataclass
class Dataconfig_openfwi_CurveVel_A24:
    size:       int   = 256
    freq:       float = 32000.# ppw = 24
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A'

@dataclass
class Dataconfig_openfwi_CurveVel_A36:
    size:       int   = 256
    freq:       float = 21333.3# ppw = 36
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/CurveVel_A/model'
    data_type:  str   = 'openfwi_CurveVel_A'



@dataclass
class Dataconfig_openfwi_FlatVel_A:
    size:       int   = 256
    freq:       float = 64000.# ppw = 12
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/FlatVel_A/model'
    data_type:  str   = 'openfwi_FlatVel_A'

@dataclass
class Dataconfig_openfwi_FlatVel_A24:
    size:       int   = 256
    freq:       float = 32000.# ppw = 24
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/FlatVel_A/model'
    data_type:  str   = 'openfwi_FlatVel_A'

@dataclass
class Dataconfig_openfwi_FlatVel_A36:
    size:       int   = 256
    freq:       float = 21333.3# ppw = 36
    dx:         float = 1./256
    data_num:   int   = 30000 
    sos_mean:   float = 3000.
    sos_max:    float = 4500.
    sos_min:    float = 1500.
    dir_path:   str   = '/afs/L202500118/wangjt/datasets/openfwi/FlatVel_A/model'
    data_type:  str   = 'openfwi_FlatVel_A'

@dataclass
class Dataconfig_openfwi3d:
    size:       int   = 192
    freq:       float = 76800.0 # ppw = 6
    dx:         float = 1./192
    data_num:   int   = 630 
    sos_mean:   int   = 2999.217642
    sos_max:    float = 4201.001465
    sos_min:    float = 2636.670410
    dir_path:   str   = '/home/wangjt/work/datasets/openfwi/3DKimberlina-V1'
    data_type:  str   = 'openfwi3d'
@dataclass
class Dataconfig_marmousi:
    size:       int   = 1024#1024#256
    freq:       float = 286.0 # ppw = 10
    dx:         float = 1.
    data_num:   int   = 1 
    sos_mean:   float = 2860.17
    sos_max:    float = 5500.
    sos_min:    float = 1500.
    dir_path:   str   = '/home/wangjt/work/datasets/Marmousi/vp.bin'
    data_type:  str   = 'marmousi'

    
def Dataconfig(user_data_type):
    if user_data_type == 'openbreastus':
        return Dataconfig_openbreastus()
    elif user_data_type == 'openbreastus6':
        return Dataconfig_openbreastus6()
    elif user_data_type == 'openbreastus8':
        return Dataconfig_openbreastus8()
    elif user_data_type == 'openbreastus10':
        return Dataconfig_openbreastus10()
    elif user_data_type == 'openbreastus12':
        return Dataconfig_openbreastus12()
    elif user_data_type == 'openbreastus14':
        return Dataconfig_openbreastus14()
    elif user_data_type == 'openbreastus16':
        return Dataconfig_openbreastus16()
    elif user_data_type == 'openfwi_CurveVel_B':
        return Dataconfig_openfwi_CurveVel_B()
    elif user_data_type == 'openfwi_CurveFault_B':
        return Dataconfig_openfwi_CurveFault_B()
    elif user_data_type == 'openfwi_CurveFault_B14':
        return Dataconfig_openfwi_CurveFault_B14()
    elif user_data_type == 'openfwi_CurveFault_B16':
        return Dataconfig_openfwi_CurveFault_B16()
    elif user_data_type == 'openfwi_CurveFault_B18':
        return Dataconfig_openfwi_CurveFault_B18()    
    elif user_data_type == 'openfwi_CurveFault_B20':
        return Dataconfig_openfwi_CurveFault_B20()
    elif user_data_type == 'openfwi_CurveFault_B24':
        return Dataconfig_openfwi_CurveFault_B24()
    elif user_data_type == 'openfwi_CurveFault_B_pad':
        return Dataconfig_openfwi_CurveFault_B_pad()

    elif user_data_type == 'openfwi_CurveVel_A':
        return Dataconfig_openfwi_CurveVel_A()
    elif user_data_type == 'openfwi_CurveVel_A14':
        return Dataconfig_openfwi_CurveVel_A14()
    elif user_data_type == 'openfwi_CurveVel_A16':
        return Dataconfig_openfwi_CurveVel_A16()
    elif user_data_type == 'openfwi_CurveVel_A20':
        return Dataconfig_openfwi_CurveVel_A20()
    elif user_data_type == 'openfwi_CurveVel_A24':
        return Dataconfig_openfwi_CurveVel_A24()
    elif user_data_type == 'openfwi_CurveVel_A36':
        return Dataconfig_openfwi_CurveVel_A36()


    elif user_data_type == 'openfwi_CurveVel_A_pad512':
        return Dataconfig_openfwi_CurveVel_A_pad512()
    elif user_data_type == 'openfwi_CurveVel_A_pad128':
        return Dataconfig_openfwi_CurveVel_A_pad128()
    elif user_data_type == 'openfwi_CurveVel_A_pad2048':
        return Dataconfig_openfwi_CurveVel_A_pad2048()


    elif user_data_type == 'openfwi_FlatVel_A':
        return Dataconfig_openfwi_FlatVel_A()
    elif user_data_type == 'openfwi_FlatVel_A24':
        return Dataconfig_openfwi_FlatVel_A24()
    elif user_data_type == 'openfwi_FlatVel_A36':
        return Dataconfig_openfwi_FlatVel_A36()
        
    elif user_data_type == 'openfwi3d':
        return Dataconfig_openfwi3d()
    elif user_data_type == 'openbreastus_inter':
        return Dataconfig_openbreastus_inter()
    elif user_data_type == 'openbreastus_pad':
        return Dataconfig_openbreastus_pad()
    elif user_data_type == 'openbreastus_pad512':
        return Dataconfig_openbreastus_pad512()
    elif user_data_type == 'openbreastus_pad1024':
        return Dataconfig_openbreastus_pad1024()
    elif user_data_type == 'openbreastus_pad2048':
        return Dataconfig_openbreastus_pad2048()
    elif user_data_type == 'openbreastus_pad4096':
        return Dataconfig_openbreastus_pad4096()
    elif user_data_type == 'openbreastus_pad8192':
        return Dataconfig_openbreastus_pad8192()
    elif user_data_type == 'marmousi':
        return Dataconfig_marmousi()
    else:
        raise NotImplementedError("no such data")
    


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    data_cfg = Dataconfig(data_type='openbreastus_inter')
    train_ds = unspvd_sos_Dataset(
        sos_path=data_cfg.dir_path,
        data_type=data_cfg.data_type,
        data_num=data_cfg.data_num,
        test=False,
    )

    loader = DataLoader(train_ds, batch_size=10, shuffle=True, num_workers=4)

    min_val = float('inf')
    max_val = float('-inf')
    mean_val = 0.0
    count = 0

    for batch in tqdm(loader, desc="Computing statistics"):
        sample = batch
        min_val = min(min_val, sample.min().item())
        max_val = max(max_val, sample.max().item())
        mean_val += sample.mean().item()
        count+=1
        
    mean_val = mean_val / count

    print(f"Mean: {mean_val:.6f}")
    print(f"Min:  {min_val:.6f}")
    print(f"Max:  {max_val:.6f}")
