import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot2Dimage(x_start=0,x_end=128,
    y_start=0,y_end=128,
    dx=1,f=None,
    cmap='RdBu_r',
    title="Wave Field",
    x_label='X',
    y_label='Y',
    figsize=(7,5),
    levels=100):
    x = torch.arange(x_start,x_end)*dx
    y = torch.arange(y_start,y_end)*dx
    X ,Y = torch.meshgrid(x,y,indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    u = f.flatten()
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.tricontourf(X,Y,u,levels=levels,cmap=cmap)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3e'))
    ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.invert_yaxis()
    plt.show()

def plot2Dimage2(x_start=0, x_end=128,
                y_start=0, y_end=128,
                dx=1, f=None,
                cmap='RdBu_r',
                title="Wave Field",
                x_label='X',
                y_label='Y',
                figsize=(7, 5),
                levels=100,
                font_size = 10):
    # 生成网格点
    x = torch.arange(x_start, x_end) * dx
    y = torch.arange(y_start, y_end) * dx
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    
    # 如果 f 是全局常量，则只显示该值并返回
    if torch.is_tensor(f) and f.numel() == 1:
        const_value = f.item()
        print(f"Global constant value for f: {const_value}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Constant Value: {const_value:.3e}", 
                fontsize=font_size, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        plt.show()
        return

    # 展平 f
    u = f.flatten()
    
    # 绘制图像
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.tricontourf(X, Y, u, levels=levels, cmap=cmap)
    
    # 设置 colorbar
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=font_size)  # 调整 colorbar 刻度字体大小
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # 降低 colorbar 刻度密度
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3e'))
    
    # 设置图像标题和轴标签字体大小
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=18)  # 调整 x 和 y 轴刻度字体大小
    ax.invert_yaxis()

    plt.show()