# Helmholtz 方程中 CBS 与 Shifted-Laplacian 预条件子的综合报告

本文总结本轮实现、调参和数值实验的完整结果。报告重点是公平比较：CBS 应在仓库原始的伪谱/FFT 离散 Helmholtz 算子上评价；有限差分 shifted-Laplacian 应在匹配的五点有限差分 Helmholtz 算子上评价。把 CBS 直接放到 FD 算子上可以作为诊断实验，但不能作为最终公平结论的主表。

## 1. 主要结论

1. **CBS 是仓库原始伪谱/FFT Helmholtz 离散的天然方法。** 在主实验 `L=2, n=160, ppw=12` 上，伪谱算子下 `FGMRES + CBS` 收敛到 `1e-6` 需要 `144` 次外迭代，`FGMRES + CBS inner(4)` 只需要 `38` 次外迭代，墙钟时间约 `0.859s`。

2. **Shifted-Laplacian 与有限差分离散更匹配。** 在 FD 算子下，FD-SL V-cycle、matrix-free FD-SL 和 sparse shifted-Laplacian LU 都能收敛。其中 sparse shifted-Laplacian LU 最强，主实验中 `54` 次外迭代、总时间约 `1.345s`。

3. **Exact FD Helmholtz LU 是直接解法，不是迭代预条件子。** 它直接分解无 shift 的 FD Helmholtz 矩阵 `A_FD`，因此 FGMRES 一步收敛。它只用于判断当前二维问题规模下直接法是否仍然可行。结果显示，在目前 `192 x 192` padded grid 规模，直接 LU 仍然很快。

4. **“inner solve” 是嵌套 Krylov 预条件步骤。** CBS inner 是在每次外层 FGMRES 预条件调用中，用固定步数的内层 FGMRES 近似求解原 Helmholtz 系统，并使用 CBS kernel 作内层预条件子。FD-SL matrix-free 的 inner solve 则是用固定步数内层 FGMRES 近似求解 shifted-Laplacian 系统。

5. **当前二维 CPU 实验还不足以体现迭代法相对直接法的规模优势。** 进一步扩大到 `L=4, n=320` 时，直接 FD LU 仍然比 FD-SL sparse LU 快。更适合体现迭代预条件优势的场景包括更大二维网格、三维问题、GPU matrix-free 计算、多右端项问题和参数扫描。

## 2. 代码实现概览

本轮主要新增或修改的文件如下：

- `helmholtz/models/cbs.py`  
  仓库原始 CBS 实现。该文件定义原始伪谱/FFT Helmholtz 算子、CBS stationary iteration 和 CBS 相关 kernel。

- `helmholtz/models/shifted_laplacian.py`  
  新增 shifted-Laplacian 系列方法：
  - `ShiftedLaplacianVcycle2d`：用于伪谱或 FD 预条件算子的几何 V-cycle 框架；
  - `FiniteDifferenceShiftedLaplacianVcycle2d`：五点 FD Helmholtz 物理算子和匹配的 FD-SL V-cycle；
  - `MatrixFreeShiftedLaplacian2d`：纯 PyTorch stencil 的 matrix-free shifted-Laplacian inner FGMRES；
  - `SparseShiftedLaplacian2d`：显式组装 shifted FD 矩阵并用 SciPy sparse LU 分解；
  - cached Vanka smoother：缓存 `2 x 2` patch 的局部逆，减少重复局部求解成本。

- `helmholtz/models/gmres.py`  
  Flexible GMRES，实现右预条件，支持预条件子在每步中变化，适合嵌套 Krylov 预条件。

- `helmholtz/compare_cbs_shifted_laplacian.py`  
  原生伪谱/FFT CBS 对比脚本。现在支持 CBS 参数调节、CBS inner Krylov、`--domain-length`、`--skip-sl`。

- `helmholtz/compare_fd_shifted_laplacian.py`  
  原生有限差分 shifted-Laplacian 对比脚本。现在支持 `--skip-cbs`，避免在最终 FD-SL 报告中混用 CBS；也支持直接 FD LU reference。

- `helmholtz/inference2d.py`  
  增加 shifted-Laplacian 类模型名的集成入口。

## 3. 连续问题与公共实验设置

考虑二维频域 Helmholtz 方程：

```text
(-Delta - omega^2 kappa(x)) u(x) = f(x),
```

其中 `kappa(x)` 是 squared slowness。实验使用确定性的中等对比度速度模型：

```text
c(x,y) = 1
       + 0.08 sin(2 pi x) cos(2 pi y)
       - 0.08 exp(-((x - 0.35)^2 + (y - 0.55)^2) / 0.025)
       + 0.07 exp(-((x - 0.72)^2 + (y - 0.38)^2) / 0.018)
```

然后

```text
kappa(x,y) = 1 / c(x,y)^2.
```

主实验设置：

```text
物理区域              [0, L]^2, L = 2
物理网格              n x n, n = 160
网格间距              dx = L / n = 0.0125
边界层                PBL
边界层宽度            16 grid cells
padded grid           192 x 192
源项                  单点源，位置约为 (n/4, n/2)
ppw                   12
频率                  mean(c) / (ppw * dx) = 6.6513574365840675
速度范围              [0.8528643755555572, 1.1332342936382955]
kappa 范围            [0.7786828018689907, 1.37480167214025]
kappa 对比度          1.7655477542851308
浮点精度              float64 / complex128
设备                  CPU
外层 Krylov 方法       FGMRES
外层相对残差容忍度     1e-6
restart               100
最大外迭代             500
```

说明：早期实验中用 `n=80, L=1`。后来根据讨论加入 `--domain-length`，使用 `n=160, L=2` 来扩大四倍物理面积，同时保持 `dx=0.0125` 和 PPW 不变。否则只增大 `n` 且令 `dx=1/n` 会变成细化同一个单位区域，而不是扩大物理域。

## 4. 外层 FGMRES、右预条件和 inner solve

所有 Krylov 实验求解离散线性系统：

```text
A u = b.
```

采用右预条件 FGMRES。预条件子应用写作：

```text
z = P(r),
```

其中 `r` 是当前 Krylov 残差方向相关向量，`P` 是某种近似逆。FGMRES 保存 `z_j = P(v_j)`，因此可以支持非固定或非严格线性的预条件子，例如内层 Krylov 预条件。

### 4.1 什么是 inner solve

“inner solve” 指每次外层 FGMRES 需要应用预条件子时，再调用一个固定预算的内层迭代解法。

#### CBS inner

`FGMRES + CBS inner(4)` 的一次预条件子应用是：

```text
给定外层向量 r:
  用 4 步内层 FGMRES 近似求解 A_PS z = r
  内层 FGMRES 的预条件子是单步 CBS kernel
  返回 z 给外层 FGMRES
```

这里内层系统仍然是原伪谱 Helmholtz 系统 `A_PS z = r`。这不是直接解法，而是固定步数的嵌套 Krylov 预条件。

#### FD-SL matrix-free inner

`FGMRES + FD-SL MF` 的一次预条件子应用是：

```text
给定外层向量 r:
  用固定步数内层 FGMRES 近似求解 M_beta z = r
  M_beta 是 shifted finite-difference Helmholtz 算子
  算子应用只使用 PyTorch stencil，不组装稀疏矩阵
  返回 z 给外层 FGMRES
```

这里内层系统是 shifted-Laplacian 系统，不是原 Helmholtz 系统。

## 5. 离散化 A：伪谱/FFT Helmholtz 与 CBS

仓库原始 CBS 使用 padded grid 上的 FFT 伪谱 Laplacian。离散算子为：

```text
A_PS u = dx^2 [ F^{-1}(|p|^2 F u) - k(x)^2 u ],
```

其中：

- `F` 是 padded grid 上的 FFT；
- `|p|^2` 是负 Laplacian 的谱符号；
- `k(x)^2 = omega^2 kappa(x)`，并包含 PBL 产生的复数吸收修正；
- FFT 本身对应 periodic padded domain，出射行为由复吸收边界近似。

### 5.1 CBS stationary iteration

CBS 定义常系数参考 Green operator：

```text
G0 = (|p|^2 - k0^2 - i epsilon)^(-1),
```

散射势：

```text
V = k(x)^2 - k0^2 - i epsilon,
```

以及松弛/缩放因子：

```text
gamma = i V / epsilon.
```

stationary CBS 更新为：

```text
u <- u - gamma [ u - G0 (V u + b / dx^2) ].
```

### 5.2 CBS as FGMRES preconditioner

CBS kernel 作为右预条件子：

```text
P_CBS(r) = gamma G0 (r / dx^2).
```

调参中允许：

```text
k0       <- k0_scale * k0
epsilon  <- epsilon_scale * epsilon
gamma    <- gamma_scale * gamma
```

主伪谱实验使用 `epsilon_scale=1.25`。在 FD 诊断调参中也发现调节 epsilon/k0 对 CBS 影响很大，说明 CBS 本身也必须公平调参。

### 5.3 FFT shifted-Laplacian V-cycle

为了在同一个伪谱算子上比较 shifted-Laplacian，也实现了 FFT shifted operator：

```text
M_beta,PS u = dx^2 [ F^{-1}(|p|^2 F u) - (1 + i beta) k(x)^2 u ].
```

V-cycle 使用：

```text
restriction       full weighting
prolongation      bilinear interpolation
smoother          damped Jacobi
coarse grid       exact coarse solve when enabled
levels            [(192,192), (96,96), (48,48), (24,24), (12,12)]
```

主伪谱实验使用 `beta=1.5`, `pre_smooth=6`, `post_smooth=6`, `num_cycles=3`。

## 6. 伪谱 CBS 原生实验

运行命令：

```bash
cd helmholtz
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python compare_cbs_shifted_laplacian.py \
  --n 160 \
  --domain-length 2 \
  --boundary-width 16 \
  --skip-cbs-stationary \
  --include-cbs-inner \
  --cbs-inner-maxiter 4 \
  --cbs-epsilon-scale 1.25 \
  --shift 1.5 \
  --sl-discretization fft \
  --pre-smooth 6 \
  --post-smooth 6 \
  --num-cycles 3 \
  --krylov-maxiter 500 \
  --out-dir ./outputs/ps_cbs_sl_n160_L2
```

图 1 给出 slowness field、解的实部、收敛曲线和墙钟时间。

![图 1：伪谱/FFT CBS 原生算子上的 CBS 与 FFT SL V-cycle 对比](outputs/ps_cbs_sl_n160_L2/cbs_vs_sl_vcycle.png)

结果表：

| 方法 | 离散算子 | 预条件子 | 外迭代 | 最终相对残差 | 墙钟时间 |
|---|---|---|---:|---:|---:|
| FGMRES + CBS | pseudo-spectral | tuned CBS kernel, `epsilon_scale=1.25` | 144 | `9.544e-07` | `2.479s` |
| FGMRES + CBS inner(4) | pseudo-spectral | 4 步内层 FGMRES + CBS kernel | 38 | `8.935e-07` | `0.859s` |
| FGMRES + SL V-cycle | pseudo-spectral | FFT shifted-Laplacian V-cycle | 334 | `9.600e-07` | `22.519s` |

原始 summary：

```text
helmholtz/outputs/ps_cbs_sl_n160_L2/summary.json
```

分析：

- CBS 与伪谱算子最匹配，因为 CBS kernel 本身就是 FFT Green operator。
- 单步 CBS kernel 已经有效；加入 4 步 inner FGMRES 后，外迭代从 `144` 降到 `38`。
- FFT SL V-cycle 虽然收敛，但 CPU 实现下每次预条件应用成本高，整体时间显著慢于 CBS。

## 7. 离散化 B：五点有限差分 Helmholtz 与 shifted-Laplacian

有限差分物理算子定义为：

```text
A_FD u = (-Delta_h u) - dx^2 k(x)^2 u.
```

五点负 Laplacian 为：

```text
(-Delta_h u)_{i,j} = 4 u_{i,j}
                   - u_{i-1,j} - u_{i+1,j}
                   - u_{i,j-1} - u_{i,j+1}.
```

这里 padded 外边界使用 zero exterior。`k(x)^2` 仍然来自仓库原 PBL 设置，因此介质和吸收层设置与 CBS 保持一致，但 Laplacian 离散不同。

### 7.1 Shifted-Laplacian operator

FD shifted-Laplacian 预条件子基于：

```text
M_beta z = r,
M_beta = (-Delta_h) - dx^2 (1 + i beta) k(x)^2.
```

复 shift `i beta k(x)^2` 增强阻尼，使多重网格或迭代近似求逆更稳定。`beta` 越小，`M_beta` 越接近原 Helmholtz 算子，但越难平滑；`beta` 越大，预条件子更稳定但与原算子距离更远。

### 7.2 FD-SL V-cycle

V-cycle 设置：

```text
coarse operators     rediscretized FD shifted operators
restriction          full weighting
prolongation         bilinear interpolation
smoother             Jacobi 或 Vanka
coarsest solve       exact coarse solve when enabled
```

最终 FD 主实验使用 Vanka smoother：

```text
beta                 0.75
levels               [(192,192), (96,96), (48,48), (24,24), (12,12)]
pre_smooth           2
post_smooth          2
num_cycles           2
Vanka patch          2 x 2
Vanka stride         2
Vanka weight         0.8
```

### 7.3 Vanka smoother

Vanka smoother 在重叠 `2 x 2` patch 上做局部 block solve。每次 sweep：

```text
1. 计算 residual r = rhs - M_beta u
2. 对每个 2 x 2 patch 提取局部 residual
3. 用局部 shifted-Laplacian block 的逆得到 patch correction
4. scatter-add 回全局网格
5. 对重叠位置取平均
6. 乘以 vanka_weight 并更新 u
```

为了避免每次 sweep 重复调用小矩阵求解，当前实现缓存局部 patch inverse，再用 batched matrix multiplication 应用 correction。这仍然是 matrix-free smoothing，因为全局稀疏矩阵没有被显式组装。

### 7.4 FD-SL matrix-free inner solve

FD-SL matrix-free 预条件子不组装 sparse matrix，而是用 PyTorch stencil 计算 `M_beta v`。一次预条件应用为：

```text
给定 r:
  用固定 maxiter 的内层 FGMRES 近似求解 M_beta z = r
  每次算子乘法都通过五点 stencil 完成
  可选 inner preconditioner: identity / Jacobi / V-cycle
  返回 z
```

最终 FD 主实验使用：

```text
beta                       0.25
inner FGMRES maxiter        12
inner preconditioner        Jacobi
```

### 7.5 FD-SL sparse LU

FD-SL sparse LU 显式组装 shifted operator：

```text
M_beta = (-Delta_h) - dx^2 (1 + i beta) k(x)^2
```

并用 SciPy sparse LU 分解。每次预条件应用通过 LU factors 解 `M_beta z = r`。注意这仍然是 shifted-Laplacian 预条件子，因为 `beta=0.25`，分解的不是原 Helmholtz 矩阵。

### 7.6 Exact FD Helmholtz LU reference

Exact FD Helmholtz LU 直接组装和分解无 shift 的原物理算子：

```text
A_FD = (-Delta_h) - dx^2 k(x)^2.
```

它应用的是 `A_FD^{-1}`。因此 FGMRES 基本一步收敛。这是直接解法参考，不是预条件方法，也不是 shifted-Laplacian 方法。

加入这个参考的目的：判断当前 2D 问题规模是否已经足够大，使迭代预条件方法超过直接稀疏分解。实验结果表明，目前规模下直接 LU 仍然非常强。

## 8. FD-SL 原生实验

该实验刻意使用 `--skip-cbs`，避免将 CBS 放到 FD 算子上作为最终主比较。

运行命令：

```bash
cd helmholtz
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python compare_fd_shifted_laplacian.py \
  --n 160 \
  --domain-length 2 \
  --boundary-width 16 \
  --skip-cbs \
  --include-identity \
  --smoother vanka \
  --vanka-stride 2 \
  --vanka-weight 0.8 \
  --pre-smooth 2 \
  --post-smooth 2 \
  --num-cycles 2 \
  --vcycle-shift 0.75 \
  --mf-shift 0.25 \
  --mf-inner-maxiter 12 \
  --mf-inner-preconditioner jacobi \
  --sparse-shift 0.25 \
  --include-fd-lu-reference \
  --krylov-maxiter 500 \
  --out-dir ./outputs/fd_sl_only_n160_L2
```

图 2 给出 FD 算子下的 slowness field、解、收敛曲线和 setup/solve 时间分解。

![图 2：FD Helmholtz 算子上的 shifted-Laplacian 系列方法](outputs/fd_sl_only_n160_L2/fd_cbs_vs_sl.png)

结果表：

| 方法 | 离散算子 | 预条件子 | 外迭代 | 最终相对残差 | setup | solve | total |
|---|---|---|---:|---:|---:|---:|---:|
| FGMRES + identity | FD | 无 | 500 | `9.616e-03` | `0.000s` | `9.505s` | `9.505s` |
| FGMRES + FD-SL V-cycle | FD | V-cycle, Vanka, `beta=0.75` | 250 | `9.811e-07` | `0.004s` | `8.566s` | `8.570s` |
| FGMRES + FD-SL MF | FD | matrix-free inner FGMRES, `beta=0.25` | 230 | `9.801e-07` | `0.003s` | `10.918s` | `10.921s` |
| FGMRES + FD-SL sparse LU | FD | shifted sparse LU, `beta=0.25` | 54 | `9.041e-07` | `0.412s` | `0.933s` | `1.345s` |
| FGMRES + FD Helmholtz LU | FD | 直接 sparse LU of `A_FD` | 1 | `1.746e-14` | `0.218s` | `0.019s` | `0.237s` |

原始 summary：

```text
helmholtz/outputs/fd_sl_only_n160_L2/summary.json
```

分析：

- 无预条件 FGMRES 到 `500` 次仍未达到 `1e-6`。
- FD-SL V-cycle 和 matrix-free FD-SL 都能收敛，但 CPU 时间较高。
- shifted sparse LU 是最强 SL 预条件子，外迭代降到 `54`。
- 直接 FD LU 仍然最快，说明当前二维问题还没有达到直接法失效的规模。

## 9. 早期 n=80 实验与调参历程

早期测试使用 `L=1, n=80, dx=0.0125`，padded grid 为 `112 x 112`。这些实验的作用是定位问题、调参并决定后续公平比较策略。

### 9.1 原始伪谱算子上的 CBS 与 SL V-cycle

图 3 是 `n=80` 上 tuned FFT SL V-cycle 与 CBS 的早期比较。

![图 3：n=80 伪谱算子上 CBS 与 FFT SL V-cycle 早期 tuned 结果](outputs/sl_vcycle_benchmark_n80_tuned/cbs_vs_sl_vcycle.png)

结果：

| 方法 | 外迭代 | 最终相对残差 | 时间 |
|---|---:|---:|---:|
| CBS stationary | 4000 | `5.758e-02` | `1.737s` |
| FGMRES + CBS | 95 | `8.836e-07` | `0.691s` |
| FGMRES + SL V-cycle | 166 | `9.846e-07` | `3.891s` |

结论：在伪谱算子上，CBS 明显更匹配。SL V-cycle 可收敛，但时间较慢。

### 9.2 FD 算子上首次综合比较

图 4 是 `n=80` FD 算子上加入 FD-SL V-cycle、matrix-free、sparse LU 后的综合结果。

![图 4：n=80 FD 算子上 FD-SL 和 CBS 诊断对比](outputs/fd_sl_comprehensive_n80_per_method/fd_cbs_vs_sl.png)

结果：

| 方法 | 外迭代 | 最终相对残差 | 总时间 |
|---|---:|---:|---:|
| FGMRES + identity | 400 | `4.210e-03` | `2.654s` |
| FGMRES + CBS applied to FD diagnostic | 172 | `9.660e-07` | `0.954s` |
| FGMRES + FD-SL V-cycle | 151 | `9.734e-07` | `2.697s` |
| FGMRES + FD-SL MF | 131 | `9.698e-07` | `2.098s` |
| FGMRES + FD-SL sparse LU | 34 | `9.579e-07` | `0.412s` |

说明：这张表中 CBS 是作为 FD 算子上的诊断性预条件子使用，不作为最终公平结论。其价值在于说明 CBS kernel 即使跨离散也有一定预条件效果，但真正公平的 CBS 结果应看伪谱算子。

### 9.3 CBS 调参

在 FD 诊断实验中，CBS 的 damping 参数对结果非常敏感。虽然这些是 FD 诊断数据，但它提醒我们：如果 SL 做高级调参，CBS 也必须调参。

| 设置 | 外迭代 | 总时间 | 说明 |
|---|---:|---:|---|
| `epsilon_scale=1.0` | 172 | `1.089s` | 默认 |
| `epsilon_scale=1.1` | 113 | `0.662s` | 明显改善 |
| `epsilon_scale=1.25` | 96 | `0.580s` | 最佳之一 |
| `epsilon_scale=1.5` | 110 | `0.695s` | 开始变差 |
| `epsilon_scale=2.0` | 136 | `0.764s` | 更差 |
| `k0_scale=1.05, epsilon_scale=1.1` | 92 | `0.540s` | 最佳 diagnostic 点 |

### 9.4 CBS inner-Krylov 调参

固定 tuned CBS kernel 后，加入内层 Krylov：

| 方法 | 外迭代 | 最终相对残差 | 总时间 |
|---|---:|---:|---:|
| CBS tuned | 92 | `8.57e-07` | `0.53-0.61s` |
| CBS inner(2) | 52 | `8.19e-07` | `0.265s` |
| CBS inner(3) | 37 | `6.98e-07` | `0.202s` |
| CBS inner(4) | 29 | `5.64e-07` | `0.198s` |
| CBS inner(6) | 19 | `6.50e-07` | `0.200s` |

结论：CBS inner 显著降低外迭代。`inner(4)` 与 `inner(6)` 时间接近；`inner(4)` 被用作主实验折中点。

### 9.5 FD-SL matrix-free 调参

Jacobi inner preconditioner 下的 matrix-free shifted solve：

| 设置 | 外迭代 | 总时间 |
|---|---:|---:|
| `beta=1.0, inner=6` | 274 | `2.564s` |
| `beta=1.0, inner=10` | 180 | `2.431s` |
| `beta=1.0, inner=12` | 153 | `2.386s` |
| `beta=1.0, inner=16` | 135 | `2.871s` |
| `beta=0.25, inner=12` | 131 | `1.985s` |
| `beta=0.5, inner=12` | 133 | `2.101s` |
| `beta=0.75, inner=12` | 139 | `2.236s` |

结论：对于 matrix-free FD-SL，较小 shift `beta=0.25` 和 `inner=12` 是较好的折中。

### 9.6 Vanka smoother 调参

Vanka smoother 的缓存实现后，FD-SL V-cycle 有所改善：

| 设置 | 外迭代 | 总时间 |
|---|---:|---:|
| `beta=1.0, stride=2, pre/post=2, cycles=2` | 152 | `2.035s` |
| `beta=1.0, stride=2, pre/post=2, cycles=3` | 129 | `2.200s` |
| `beta=0.75, stride=2, pre/post=2, cycles=2` | 141 | `1.784s` |
| `beta=0.5, stride=2, pre/post=2, cycles=2` | 193 | `2.569s` |
| `beta=1.25, stride=2, pre/post=2, cycles=2` | 179 | `2.412s` |

结论：Vanka 对 V-cycle 有帮助，最佳测试点约为 `beta=0.75, cycles=2`。不过 CPU 时间仍然高于 sparse LU 和 tuned CBS diagnostic。

## 10. 放大域与 scaling 诊断

为了回应“总波数太小、需要放大物理区域”的问题，脚本增加了 `--domain-length`。这样可以保持 `dx` 和 PPW 不变，同时扩大物理区域。

### 10.1 `L=2, n=160`

这是主实验。相对原始 `L=1, n=80`，物理面积变为 4 倍，`dx` 保持 `0.0125`。

### 10.2 `L=4, n=320`

进一步测试：

```text
L = 4
n = 320
dx = 0.0125
padded grid = 352 x 352
```

图 5 是该 scaling diagnostic。

![图 5：L=4, n=320 放大域 scaling diagnostic](outputs/fd_scaling_n320_L4_key_methods/fd_cbs_vs_sl.png)

结果：

| 方法 | 外迭代 | 最终相对残差 | 总时间 |
|---|---:|---:|---:|
| FGMRES + CBS tuned applied to FD diagnostic | 264 | `9.696e-07` | `11.934s` |
| FGMRES + CBS inner(4) applied to FD diagnostic | 71 | `9.537e-07` | `4.696s` |
| FGMRES + FD-SL sparse LU | 94 | `9.788e-07` | `7.940s` |
| FGMRES + FD Helmholtz LU | 1 | `2.790e-14` | `1.399s` |

原始 summary：

```text
helmholtz/outputs/fd_scaling_n320_L4_key_methods/summary.json
```

结论：即使到 `352 x 352` padded grid，二维单右端项问题上直接 FD LU 仍然很强。要看到迭代法明显超过直接法，可能需要更大的二维问题，或者三维问题。

## 11. 为什么最终报告分开伪谱和 FD 两类实验

CBS 和 FD-SL 都在求解连续 Helmholtz 问题的某种离散形式，但离散矩阵不同：

```text
A_PS != A_FD.
```

因此，不能把 `A_PS` 上的 CBS 残差和 `A_FD` 上的 FD-SL 残差放进同一个最终排行榜，声称谁绝对更优。公平做法是：

1. 在 `A_PS` 上比较 CBS 与使用 FFT Laplacian 的 SL；
2. 在 `A_FD` 上比较 FD-SL 的各种实现；
3. 若做跨离散诊断，必须明确标注为 diagnostic，而不是最终公平比较。

这个原则也是本报告最终结构的核心。

## 12. 重要结论与后续建议

### 12.1 对 CBS 的结论

- CBS 应继续用于仓库原始伪谱/FFT 离散。
- CBS kernel 与伪谱算子天然匹配，单步预条件已经有效。
- CBS inner-Krylov 是非常有效的增强：主伪谱实验中 `38` 次外迭代、`0.859s`。
- CBS 也需要调参；特别是 `epsilon_scale` 会显著影响收敛。

### 12.2 对 shifted-Laplacian 的结论

- SL 与 FD 离散更兼容。
- V-cycle + Vanka 能收敛，但当前 CPU 实现每次预条件成本仍高。
- Matrix-free FD-SL 纯 PyTorch stencil 实现是正确的，也能收敛，但 inner Krylov 成本较高。
- Sparse shifted-Laplacian LU 是目前最强的 SL 变体，但它依赖 sparse factorization。

### 12.3 对直接 LU reference 的结论

- Exact FD Helmholtz LU 是直接解法，不是预条件子。
- 在当前二维单 RHS 规模，它最快。
- 这说明当前测试规模尚未进入直接法不可承受的区间。

### 12.4 后续建议

1. 做三维 FD Helmholtz 测试。三维中直接 LU 的内存和时间会迅速增长，迭代预条件更有意义。
2. 做更大二维网格，直到 sparse LU 的内存成为瓶颈。
3. 在 GPU 上测试 matrix-free SL 和 CBS inner。
4. 多 RHS 或频率扫描时，区分 setup cost 和 solve cost 的摊销。
5. 继续实现更先进的 SL 多重网格技术，例如更强的 block smoother、Kaczmarz/SOR-like smoother、operator-dependent interpolation 或 deflation/coarse-space correction。
6. 若要在伪谱算子上发展 SL，应设计真正匹配 FFT/pseudo-spectral discretization 的 SL 多层策略，而不是直接照搬 FD multigrid。

## 13. 复现实验命令汇总

伪谱 CBS 原生主实验：

```bash
cd helmholtz
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python compare_cbs_shifted_laplacian.py \
  --n 160 --domain-length 2 --boundary-width 16 \
  --skip-cbs-stationary \
  --include-cbs-inner --cbs-inner-maxiter 4 \
  --cbs-epsilon-scale 1.25 \
  --shift 1.5 --sl-discretization fft \
  --pre-smooth 6 --post-smooth 6 --num-cycles 3 \
  --krylov-maxiter 500 \
  --out-dir ./outputs/ps_cbs_sl_n160_L2
```

FD-SL 原生主实验：

```bash
cd helmholtz
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python compare_fd_shifted_laplacian.py \
  --n 160 --domain-length 2 --boundary-width 16 \
  --skip-cbs --include-identity \
  --smoother vanka --vanka-stride 2 --vanka-weight 0.8 \
  --pre-smooth 2 --post-smooth 2 --num-cycles 2 \
  --vcycle-shift 0.75 \
  --mf-shift 0.25 --mf-inner-maxiter 12 --mf-inner-preconditioner jacobi \
  --sparse-shift 0.25 \
  --include-fd-lu-reference \
  --krylov-maxiter 500 \
  --out-dir ./outputs/fd_sl_only_n160_L2
```

放大域 scaling diagnostic：

```bash
cd helmholtz
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python compare_fd_shifted_laplacian.py \
  --n 320 --domain-length 4 --boundary-width 16 \
  --skip-vcycle --skip-matrix-free \
  --include-cbs-inner --cbs-inner-maxiter 4 \
  --cbs-k0-scale 1.05 --cbs-epsilon-scale 1.1 \
  --sparse-shift 0.25 \
  --include-fd-lu-reference \
  --krylov-maxiter 500 \
  --out-dir ./outputs/fd_scaling_n320_L4_key_methods
```

## 14. 验证

代码编译检查：

```bash
/Users/hxchen/miniconda3-arm64/envs/py312_env/bin/python -m py_compile \
  helmholtz/models/shifted_laplacian.py \
  helmholtz/compare_fd_shifted_laplacian.py \
  helmholtz/compare_cbs_shifted_laplacian.py \
  helmholtz/verify_shifted_laplacian.py \
  helmholtz/inference2d.py
```

检查通过。

