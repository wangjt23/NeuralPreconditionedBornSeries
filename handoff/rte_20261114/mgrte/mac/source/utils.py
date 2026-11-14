import torch
import os
import time

try:
    import gpustat
except ImportError:  # gpustat is only needed for GPU auto-selection.
    gpustat = None


def get_least_used_gpu():
    if gpustat is None:
        raise RuntimeError("gpustat is required for get_least_used_gpu()")
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU

def check_cuda_mem(device):
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        return
    print('>>> memory usage: %.3f GB' % (torch.cuda.memory_allocated(device) / 1024**3))
    print('>>> memory cached: %.3f GB' % (torch.cuda.memory_reserved(device) / 1024**3))

def printkeyval(key, value):
    # print key with extra spaces to align values
    print(key + ' ' * (25 - len(key)), value)

def greenprint(print_str):
    print('\033[92m%s\033[0m' % print_str)
    
def redprint(print_str):
    print('\033[91m%s\033[0m' % print_str)

def yellowprint(print_str):
    print('\033[93m%s\033[0m' % print_str)
    
def blueprint(print_str):
    print('\033[94m%s\033[0m' % print_str)

def save_loss_history(dir_path, loss_list, fn):
    import matplotlib.pyplot as plt

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, '%s.txt' % fn), 'w') as f:
        for loss in loss_list:
            f.write('%.5e\n' % loss)
    if loss_list:
        plt.semilogy(loss_list)
    plt.savefig(os.path.join(dir_path, '%s.png' % fn))
    plt.close()
    print("loss list saved!")
    
def save_model(dir_path, model, fn):
    """
    Save a PyTorch model's state_dict to a specified directory with filename prefix `fn`.

    Args:
        dir_path (str): Directory to save the model.
        model (torch.nn.Module): The trained PyTorch model.
        fn (str): Filename prefix.
    """
    # Check if directory exists
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    
    # Save model parameters
    save_path = os.path.join(dir_path, f'{fn}.pt')
    torch.save(model.state_dict(), save_path)
    print("model saved!")
    


class Timer():
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and torch.cuda.is_available():
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = time.time()
    def reset(self):
        if self.device.type == 'cuda' and torch.cuda.is_available():
            self.start_time.record()
        else:
            self.start_time = time.time()
    def elapsed(self, type='ms'):
        elasped_time = 0
        if self.device.type == 'cuda' and torch.cuda.is_available():
            self.end_time.record()
            torch.cuda.synchronize()
            elasped_time = self.start_time.elapsed_time(self.end_time)
        else:
            elasped_time = time.time() - self.start_time
        if type == 'ms':
            return elasped_time
        elif type == 's':
            return elasped_time / 1000 if self.device.type == 'cuda' and torch.cuda.is_available() else elasped_time
        elif type == 'min':
            return (elasped_time / 1000 if self.device.type == 'cuda' and torch.cuda.is_available() else elasped_time) / 60
        else:
            return elasped_time


def check_chinese_font():
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # 下载思源黑体（首次运行需要下载）
    font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
    font_path = "SourceHanSansSC-Regular.otf"
    # 如果字体文件不存在，下载
    if not os.path.exists(font_path):
        import urllib.request
        print("正在下载中文字体...")
        urllib.request.urlretrieve(font_url, font_path)
        print("下载完成！")
    # 添加字体
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Source Han Sans SC'
    plt.rcParams['axes.unicode_minus'] = False
