import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import sqlite3
import os


# ======================== 基础数据（IEEE 9-bus） ========================

baseMVA = 100.0

# bus: [BUS_I, BUS_TYPE(3=slack,2=PV,1=PQ), Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
bus = np.array([
    [1, 3, 0, 0, 0, 0, 1, 1.04, 0.0, 345, 1, 1.1, 0.9],
    [2, 2, 0, 0, 0, 0, 1, 1.025, 0.0, 345, 1, 1.1, 0.9],
    [3, 2, 0, 0, 0, 0, 1, 1.025, 0.0, 345, 1, 1.1, 0.9],
    [4, 1, 0, 0, 0, 0, 1, 1.0,   0.0, 345, 1, 1.1, 0.9],
    [5, 1, 54, 18, 0, 0, 1, 1.0, 0.0, 345, 1, 1.1, 0.9],
    [6, 1, 0, 0, 0, 0, 1, 1.0,   0.0, 345, 1, 1.1, 0.9],
    [7, 1, 60, 21, 0, 0, 1, 1.0, 0.0, 345, 1, 1.1, 0.9],
    [8, 1, 0, 0, 0, 0, 1, 1.0,   0.0, 345, 1, 1.1, 0.9],
    [9, 1, 75, 30, 0, 0, 1, 1.0, 0.0, 345, 1, 1.1, 0.9],
], dtype=float)

# gen: [BUS, Pg(MW), Qg(MVAr), Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, ...]
# Bus1(slack), Bus2/3(PV)
gen = np.array([
    [1,   0, 0,  300,  -5, 1.04, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 163, 0,  300,  -5, 1.025,100,1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3,  85, 0,  300,  -5, 1.025,100,1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float)

# branch: [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]
branch = np.array([
    [1, 4, 0.0000, 0.0576, 0.0000, 250, 250, 250, 0, 0, 1, -360, 360],
    [4, 5, 0.0170, 0.0920, 0.1580, 250, 250, 250, 0, 0, 1, -360, 360],
    [5, 6, 0.0390, 0.1700, 0.3580, 150, 150, 150, 0, 0, 1, -360, 360],
    [3, 6, 0.0000, 0.0586, 0.0000, 300, 300, 300, 0, 0, 1, -360, 360],
    [6, 7, 0.0119, 0.1008, 0.2090, 150, 150, 150, 0, 0, 1, -360, 360],
    [7, 8, 0.0085, 0.0720, 0.1490, 250, 250, 250, 0, 0, 1, -360, 360],
    [8, 2, 0.0000, 0.0625, 0.0000, 250, 250, 250, 0, 0, 1, -360, 360],
    [8, 9, 0.0320, 0.1610, 0.3060, 250, 250, 250, 0, 0, 1, -360, 360],
    [9, 4, 0.0100, 0.0850, 0.1760, 250, 250, 250, 0, 0, 1, -360, 360],
], dtype=float)


# ======================== 工具与系统构造（向量化 + 设备/精度控制） ========================

def build_ybus(n_bus: int, branch: np.ndarray) -> np.ndarray:
    Ybus = np.zeros((n_bus, n_bus), dtype=complex)
    for br in branch:
        f = int(br[0]) - 1
        t = int(br[1]) - 1
        r, x, b = br[2:5]
        z = complex(r, x)
        if abs(z) < 1e-12:
            y = 1 / complex(0, x)
        else:
            y = 1 / z
        b_sh = 1j * b / 2.0
        Ybus[f, f] += y + b_sh
        Ybus[t, t] += y + b_sh
        Ybus[f, t] -= y
        Ybus[t, f] -= y
    return Ybus


@dataclass
class CaseData:
    n_bus: int
    n_gen: int
    slack_idx: int
    pv_idx: List[int]
    pq_idx: List[int]
    Pd: torch.Tensor  # p.u.
    Qd: torch.Tensor  # p.u.
    Vmin: torch.Tensor
    Vmax: torch.Tensor
    Vset: torch.Tensor
    gen_bus: List[int]
    Pg_given: torch.Tensor  # p.u., for PV buses; slack ignored
    Qg_min: torch.Tensor    # p.u., for all generators
    Qg_max: torch.Tensor    # p.u., for all generators
    YG: torch.Tensor
    YB: torch.Tensor
    device: torch.device
    dtype: torch.dtype


def prepare_case(load_factor: float = 1.0,
                 pv_pg_MW: Optional[Dict[int, float]] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32) -> CaseData:
    """
    pv_pg_MW: 可选，键为母线号(1-based)，值为该PV母线机组的给定有功(MW)。
              仅对 PV 机组有效（slack无效）。
    """
    dev = torch.device(device)
    n_bus = bus.shape[0]
    n_gen = gen.shape[0]
    # bus types
    slack_idx = int(np.where(bus[:, 1] == 3)[0][0])
    pv_idx = [int(i) for i in np.where(bus[:, 1] == 2)[0].tolist()]
    pq_idx = [int(i) for i in np.where(bus[:, 1] == 1)[0].tolist()]

    # loads (p.u.)
    Pd = torch.tensor((bus[:, 2] / baseMVA) * load_factor, dtype=dtype, device=dev)
    Qd = torch.tensor((bus[:, 3] / baseMVA) * load_factor, dtype=dtype, device=dev)

    # voltage bounds and setpoints
    Vmin = torch.tensor(bus[:, 12], dtype=dtype, device=dev)
    Vmax = torch.tensor(bus[:, 11], dtype=dtype, device=dev)
    Vset = torch.tensor(bus[:, 7], dtype=dtype, device=dev)

    # generators
    gen_bus = [int(g[0]) - 1 for g in gen]
    Pg_given = torch.tensor(gen[:, 1] / baseMVA, dtype=dtype, device=dev)
    Qg_max = torch.tensor(gen[:, 3] / baseMVA, dtype=dtype, device=dev)
    Qg_min = torch.tensor(gen[:, 4] / baseMVA, dtype=dtype, device=dev)

    # 覆盖 PV 给定有功（以母线号 1-based 指定）
    if pv_pg_MW is not None:
        for gi, b in enumerate(gen_bus):
            if b != slack_idx:  # 仅 PV
                bus_no = b + 1  # 1-based
                if bus_no in pv_pg_MW:
                    Pg_given[gi] = float(pv_pg_MW[bus_no]) / baseMVA

    # Ybus
    Y = build_ybus(n_bus, branch)
    YG = torch.tensor(Y.real, dtype=dtype, device=dev)
    YB = torch.tensor(Y.imag, dtype=dtype, device=dev)

    return CaseData(
        n_bus=n_bus,
        n_gen=n_gen,
        slack_idx=slack_idx,
        pv_idx=pv_idx,
        pq_idx=pq_idx,
        Pd=Pd, Qd=Qd,
        Vmin=Vmin, Vmax=Vmax, Vset=Vset,
        gen_bus=gen_bus,
        Pg_given=Pg_given,
        Qg_min=Qg_min, Qg_max=Qg_max,
        YG=YG, YB=YB,
        device=dev, dtype=dtype
    )


# ======================== H(x) 构造（2.3/2.4，向量化） ========================

@dataclass
class TJUStateIndex:
    idx_Va: slice
    idx_Vm: slice
    idx_Pg_slack: slice
    idx_sv_over: slice
    idx_sv_under: slice
    idx_sq_over: slice
    idx_sq_under: slice
    n_vars: int


@dataclass
class TJUModel:
    data: CaseData
    idx: TJUStateIndex


def build_state_index(data: CaseData) -> TJUStateIndex:
    n = data.n_bus
    p = 0
    idx_Va = slice(p, p + n); p += n
    idx_Vm = slice(p, p + n); p += n
    idx_Pg_slack = slice(p, p + 1); p += 1
    idx_sv_over = slice(p, p + n); p += n
    idx_sv_under = slice(p, p + n); p += n
    idx_sq_over = slice(p, p + data.n_gen); p += data.n_gen
    idx_sq_under = slice(p, p + data.n_gen); p += data.n_gen
    return TJUStateIndex(idx_Va, idx_Vm, idx_Pg_slack, idx_sv_over, idx_sv_under, idx_sq_over, idx_sq_under, p)


def tju_build_model(load_factor: float = 1.0,
                    pv_pg_MW: Optional[Dict[int, float]] = None,
                    device: str = "cpu",
                    dtype: torch.dtype = torch.float32) -> TJUModel:
    data = prepare_case(load_factor=load_factor, pv_pg_MW=pv_pg_MW, device=device, dtype=dtype)
    idx = build_state_index(data)
    return TJUModel(data, idx)


def power_injections(Vm: torch.Tensor, Va: torch.Tensor,
                     YG: torch.Tensor, YB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    向量化的注入计算（GPU友好）：
    Pi = sum_j |Vi||Vj|(G_ij*cos θij + B_ij*sin θij)
    Qi = sum_j |Vi||Vj|(G_ij*sin θij - B_ij*cos θij)
    """
    theta = Va.unsqueeze(1) - Va.unsqueeze(0)   # [n,n]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    ViVj = Vm.unsqueeze(1) * Vm.unsqueeze(0)   # [n,n]
    Pi = torch.sum(ViVj * (YG * cos_t + YB * sin_t), dim=1)
    Qi = torch.sum(ViVj * (YG * sin_t - YB * cos_t), dim=1)
    return Pi, Qi


def assemble_H(x: torch.Tensor, mdl: TJUModel) -> torch.Tensor:
    d = mdl.data
    ix = mdl.idx

    Va = x[ix.idx_Va]
    Vm = x[ix.idx_Vm]
    Pg1 = x[ix.idx_Pg_slack].squeeze()

    sV_over = x[ix.idx_sv_over]
    sV_under = x[ix.idx_sv_under]
    sQ_over = x[ix.idx_sq_over]
    sQ_under = x[ix.idx_sq_under]

    # 网络注入
    Pi, Qi = power_injections(Vm, Va, d.YG, d.YB)

    # 生成各母线发电有功 Pg_i（p.u.）
    Pg = torch.zeros(d.n_bus, dtype=d.dtype, device=d.device)
    for gi, b in enumerate(d.gen_bus):
        if b == d.slack_idx:
            Pg[b] = Pg1
        else:
            Pg[b] = d.Pg_given[gi]

    H_list: List[torch.Tensor] = []

    # 1) 有功平衡（所有母线）
    H_P = Pg - d.Pd - Pi
    H_list.append(H_P)

    # 2) 无功平衡（PQ母线）
    if len(d.pq_idx):
        H_Q = torch.stack([(- d.Qd[i] - Qi[i]) for i in d.pq_idx])
        H_list.append(H_Q)

    # 3) 电压设定等式：Slack 与 PV 的 Vm_i = Vset_i
    set_idx = [d.slack_idx] + d.pv_idx
    if len(set_idx):
        H_Vset = torch.stack([Vm[i] - d.Vset[i] for i in set_idx])
        H_list.append(H_Vset)

    # 4) 参考角：Va_slack = 0
    H_va0 = Va[d.slack_idx].unsqueeze(0)
    H_list.append(H_va0)

    # 5) 不等式 —— 电压上下限（松弛平方等式）
    H_v_over = Vm - d.Vmax + sV_over**2
    H_v_under = d.Vmin - Vm + sV_under**2
    H_list.append(H_v_over)
    H_list.append(H_v_under)

    # 6) 不等式 —— 发电机无功上下限（Slack+PV）
    H_q_over = []
    H_q_under = []
    for gi, b in enumerate(d.gen_bus):
        Qg_i = d.Qd[b] + Qi[b]
        H_q_over.append(Qg_i - d.Qg_max[gi] + sQ_over[gi]**2)
        H_q_under.append(d.Qg_min[gi] - Qg_i + sQ_under[gi]**2)
    H_list.append(torch.stack(H_q_over))
    H_list.append(torch.stack(H_q_under))

    H = torch.cat(H_list)
    return H


# ======================== 动力学法（QGS）与分类 ========================

@dataclass
class TJUOptions:
    max_steps: int = 20000
    tol_res: float = 1e-6      # 放宽默认容差，加速收敛（原1e-9较严）
    dt_init: float = 0.05
    dt_inc: float = 1.05
    dt_dec: float = 0.5
    dt_max: float = 1.0
    verbose: bool = False      # 开启时打印迭代日志
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "auto"        # "auto"=>cuda用float32, cpu用float64；或显式 "float32"/"float64"
    seed: int = 42
    print_every: int = 100     # 迭代日志频率
    max_time: float = 3.0      # 单点最大迭代时间（秒），避免个别点拖慢全局


@dataclass
class TJUResult:
    status: str           # "RSEP" / "DSEP" / "MAXSTEP" / "MAXTIME"
    steps: int
    time: float
    res_norm: float
    Pg1: float            # MW
    Vm: np.ndarray
    Va: np.ndarray
    Qg: np.ndarray        # p.u. (gens)
    max_mismatch: float
    message: str


def get_torch_dtype(name: str, device: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    # auto
    if device.startswith("cuda"):
        return torch.float32  # GPU上 float32 更快
    else:
        return torch.float64  # CPU上保留双精度稳定性


def init_state(mdl: TJUModel) -> torch.Tensor:
    d = mdl.data
    ix = mdl.idx
    x = torch.zeros(ix.n_vars, dtype=d.dtype, device=d.device, requires_grad=True)

    with torch.no_grad():
        # Va 初值
        x[ix.idx_Va] = torch.zeros(d.n_bus, dtype=d.dtype, device=d.device)
        # Vm 初值（界内，尽量贴近设定）
        Vm0 = torch.clamp(d.Vset, d.Vmin, d.Vmax)
        x[ix.idx_Vm] = Vm0
        # Pg_slack 初值：总负荷 - PV给定
        total_Pd = d.Pd.sum()
        Pg_pv = 0.0
        for gi, b in enumerate(d.gen_bus):
            if b != d.slack_idx:
                Pg_pv += d.Pg_given[gi]
        # 使用 gen 数据中的 Pmin/Pmax 进行夹取
        P1min = torch.tensor(gen[0, 9] / baseMVA, dtype=d.dtype, device=d.device)
        P1max = torch.tensor(gen[0, 8] / baseMVA, dtype=d.dtype, device=d.device)
        Pg1_0 = torch.clamp(total_Pd - Pg_pv, min=P1min, max=P1max)
        x[ix.idx_Pg_slack] = Pg1_0

        # 电压不等式松弛初值
        x[ix.idx_sv_over] = torch.sqrt(torch.clamp(d.Vmax - Vm0, min=0.0))
        x[ix.idx_sv_under] = torch.sqrt(torch.clamp(Vm0 - d.Vmin, min=0.0))

        # Qg不等式松弛初值
        _, Qi = power_injections(Vm0, x[ix.idx_Va], d.YG, d.YB)
        sQ_over = torch.zeros(d.n_gen, dtype=d.dtype, device=d.device)
        sQ_under = torch.zeros(d.n_gen, dtype=d.dtype, device=d.device)
        for gi, b in enumerate(d.gen_bus):
            Qg_i = d.Qd[b] + Qi[b]
            sQ_over[gi] = torch.sqrt(torch.clamp(d.Qg_max[gi] - Qg_i, min=0.0))
            sQ_under[gi] = torch.sqrt(torch.clamp(Qg_i - d.Qg_min[gi], min=0.0))
        x[ix.idx_sq_over] = sQ_over
        x[ix.idx_sq_under] = sQ_under

    x.requires_grad_(True)
    return x


def energy_and_grad(xx: torch.Tensor, mdl: TJUModel) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H = assemble_H(xx, mdl)
    E = 0.5 * torch.dot(H, H)
    g = torch.autograd.grad(E, xx, create_graph=False)[0]
    res = torch.linalg.vector_norm(H)
    return E, g, res


def component_norms(xx: torch.Tensor, mdl: TJUModel) -> Dict[str, float]:
    """用于 DSEP 诊断：分组显示 H(x) 各组件的 L2 范数"""
    d = mdl.data
    ix = mdl.idx
    Va = xx[ix.idx_Va]
    Vm = xx[ix.idx_Vm]
    Pg1 = xx[ix.idx_Pg_slack].squeeze()
    sV_over = xx[ix.idx_sv_over]
    sV_under = xx[ix.idx_sv_under]
    sQ_over = xx[ix.idx_sq_over]
    sQ_under = xx[ix.idx_sq_under]

    Pi, Qi = power_injections(Vm, Va, d.YG, d.YB)
    Pg = torch.zeros(d.n_bus, dtype=d.dtype, device=d.device)
    for gi, b in enumerate(d.gen_bus):
        Pg[b] = Pg1 if b == d.slack_idx else d.Pg_given[gi]

    # 组建分量
    H_P = Pg - d.Pd - Pi
    H_Q = torch.stack([(- d.Qd[i] - Qi[i]) for i in d.pq_idx]) if len(d.pq_idx) else torch.zeros(0, dtype=d.dtype, device=d.device)
    H_Vset = torch.stack([Vm[i] - d.Vset[i] for i in [d.slack_idx] + d.pv_idx]) if (len(d.pv_idx)+1) else torch.zeros(0, dtype=d.dtype, device=d.device)
    H_va0 = Va[d.slack_idx].unsqueeze(0)
    H_v_over = Vm - d.Vmax + sV_over**2
    H_v_under = d.Vmin - Vm + sV_under**2
    H_q_over = []
    H_q_under = []
    for gi, b in enumerate(d.gen_bus):
        Qg_i = d.Qd[b] + Qi[b]
        H_q_over.append(Qg_i - d.Qg_max[gi] + sQ_over[gi]**2)
        H_q_under.append(d.Qg_min[gi] - Qg_i + sQ_under[gi]**2)
    H_q_over = torch.stack(H_q_over)
    H_q_under = torch.stack(H_q_under)

    norms = {
        "P-balance": float(torch.linalg.vector_norm(H_P).item()),
        "Q-balance(PQ)": float(torch.linalg.vector_norm(H_Q).item()) if H_Q.numel() else 0.0,
        "V-set(Slack+PV)": float(torch.linalg.vector_norm(H_Vset).item()) if H_Vset.numel() else 0.0,
        "Va_slack": float(torch.linalg.vector_norm(H_va0).item()),
        "V<=Vmax": float(torch.linalg.vector_norm(H_v_over).item()),
        "V>=Vmin": float(torch.linalg.vector_norm(H_v_under).item()),
        "Qg<=Qmax": float(torch.linalg.vector_norm(H_q_over).item()),
        "Qg>=Qmin": float(torch.linalg.vector_norm(H_q_under).item()),
    }
    return norms


def classify_point(x: torch.Tensor, mdl: TJUModel, res_norm: float, tol: float) -> Tuple[str, float]:
    # 简化判定：‖H‖ < tol 即 RSEP，否则 DSEP，max_mismatch 用等式部分度量
    d = mdl.data
    with torch.no_grad():
        Va = x[mdl.idx.idx_Va]
        Vm = x[mdl.idx.idx_Vm]
        Pg1 = x[mdl.idx.idx_Pg_slack].squeeze()
        Pi, Qi = power_injections(Vm, Va, d.YG, d.YB)
        Pg = torch.zeros(d.n_bus, dtype=d.dtype, device=d.device)
        for gi, b in enumerate(d.gen_bus):
            Pg[b] = Pg1 if b == d.slack_idx else d.Pg_given[gi]

        parts = [Pg - d.Pd - Pi]
        if len(d.pq_idx):
            parts.append(torch.stack([(- d.Qd[i] - Qi[i]) for i in d.pq_idx]))
        if (len(d.pv_idx) + 1):
            parts.append(torch.stack([Vm[i] - d.Vset[i] for i in [d.slack_idx] + d.pv_idx]))
        parts.append(Va[d.slack_idx].unsqueeze(0))
        eq_vec = torch.cat([p.flatten() for p in parts]) if len(parts) else torch.tensor([res_norm], dtype=d.dtype, device=d.device)
        max_mis = float(torch.max(torch.abs(eq_vec)).item())

    if res_norm < tol:
        return "RSEP", max_mis
    else:
        return "DSEP", max_mis


def tju_solve(load_factor: float = 1.0,
              opts: TJUOptions = TJUOptions(),
              pv_pg_MW: Optional[Dict[int, float]] = None) -> TJUResult:
    """
    pv_pg_MW: {2:P2_MW, 3:P3_MW} 这样的字典（母线号为1-based）。
    """
    torch.manual_seed(opts.seed)
    device = opts.device
    dtype = get_torch_dtype(opts.dtype, device)

    mdl = tju_build_model(load_factor=load_factor, pv_pg_MW=pv_pg_MW, device=device, dtype=dtype)
    x = init_state(mdl).detach().clone().requires_grad_(True)

    dt = opts.dt_init
    E, g, res = energy_and_grad(x, mdl)
    start = time.time()
    step = 0

    # 迭代日志频率
    PRINT_EVERY = max(1, int(opts.print_every))
    # 能量未降低的连续尝试计数（防止频繁回退造成长时间卡滞）
    reject_count = 0
    REJECT_LIMIT = 20

    while True:
        step += 1
        # 超时保护
        if time.time() - start > opts.max_time:
            status = "MAXTIME"
            break
        if step > opts.max_steps:
            status = "MAXSTEP"
            break

        with torch.no_grad():
            x_new = (x - dt * g).detach().clone().requires_grad_(True)

        E_new, g_new, res_new = energy_and_grad(x_new, mdl)

        if E_new <= E:
            # 接受步
            x = x_new
            E = E_new
            g = g_new
            res = res_new
            dt = min(opts.dt_max, dt * opts.dt_inc)
            reject_count = 0

            # 迭代日志（仅接受步打印）
            if opts.verbose and (step % PRINT_EVERY == 0 or step in (1, 5, 10)):
                with torch.no_grad():
                    Pg1_MW = float(x[mdl.idx.idx_Pg_slack].item() * baseMVA)
                print(f"[iter {step:6d}] E={E.item():.3e}  ||H||={res.item():.3e}  dt={dt:.3e}  P1={Pg1_MW:.2f} MW")
                # 可选：打印主要分量范数
                try:
                    norms = component_norms(x, mdl)
                    print("           H parts: "
                          f"P={norms['P-balance']:.2e}, "
                          f"Q={norms['Q-balance(PQ)']:.2e}, "
                          f"Vset={norms['V-set(Slack+PV)']:.2e}, "
                          f"Va0={norms['Va_slack']:.2e}")
                except Exception:
                    pass
        else:
            # 回退减步；若连续过多次拒绝，直接判DSEP退出
            dt = max(1e-8, dt * opts.dt_dec)
            reject_count += 1
            if reject_count >= REJECT_LIMIT:
                status = "DSEP"
                break
            continue  # 回退减步后重试

        if float(res.item()) < opts.tol_res:
            status = "RSEP"
            break

    elapsed = time.time() - start

    # 结果解析
    with torch.no_grad():
        Va = x[mdl.idx.idx_Va].detach().cpu().numpy()
        Vm = x[mdl.idx.idx_Vm].detach().cpu().numpy()
        Pg1 = float(x[mdl.idx.idx_Pg_slack].item() * baseMVA)
        _, Qi = power_injections(x[mdl.idx.idx_Vm], x[mdl.idx.idx_Va], mdl.data.YG, mdl.data.YB)
        Qg = []
        for gi, b in enumerate(mdl.data.gen_bus):
            Qg.append(float((mdl.data.Qd[b] + Qi[b]).item()))
        Qg = np.array(Qg)

    status2, max_mismatch = classify_point(x, mdl, float(res.item()), opts.tol_res)
    # 以更严格的分类覆盖 DSEP/MAXSTEP 的情况（若实际可行则修正为 RSEP）
    if status != "RSEP" and status2 == "RSEP":
        status = "RSEP"

    if status == "RSEP":
        msg = "RSEP（可行潮流解）：||H|| 已小于阈值。"
    elif status == "MAXTIME":
        msg = f"达到单点最大计算时长 {opts.max_time:.1f}s，已停止。"
    elif status == "MAXSTEP":
        msg = f"达到最大步数 {opts.max_steps}，已停止。"
    else:
        # DSEP 简要诊断
        norms = component_norms(x, mdl)
        worst = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)[:3]
        tops = ", ".join([f"{k}:{v:.2e}" for k, v in worst])
        msg = f"DSEP（退化点）：最大等式不匹配≈{max_mismatch:.2e}；主导分量 ≈ {tops}"

    return TJUResult(
        status=status,
        steps=step,
        time=elapsed,
        res_norm=float(res.item()),
        Pg1=Pg1,
        Vm=Vm,
        Va=Va,
        Qg=Qg,
        max_mismatch=max_mismatch,
        message=msg
    )


# ======================== 数据库落盘（释放内存） ========================

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    p2 REAL,
    p3 REAL,
    status TEXT,
    p1 REAL,
    res_norm REAL,
    max_mismatch REAL,
    steps INTEGER,
    time REAL,
    note TEXT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

DB_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_status ON scan_results(status);",
    "CREATE INDEX IF NOT EXISTS idx_p2p3 ON scan_results(p2, p3);"
]

def init_database(db_path: str, reset: bool = True):
    if reset and os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(DB_SCHEMA)
    for stmt in DB_INDEXES:
        cur.execute(stmt)
    conn.commit()
    return conn


def insert_result(cur: sqlite3.Cursor,
                  p2: float, p3: float,
                  ret: TJUResult):
    cur.execute("""
        INSERT INTO scan_results (p2, p3, status, p1, res_norm, max_mismatch, steps, time, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (float(p2), float(p3), ret.status, (ret.Pg1 if ret.status == "RSEP" else None),
          float(ret.res_norm), float(ret.max_mismatch), int(ret.steps), float(ret.time), ret.message))


def export_csv_from_db(db_path: str, csv_path: str):
    import csv
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT p2, p3, status, p1, res_norm, max_mismatch, steps, time, note, ts FROM scan_results ORDER BY p2, p3;")
    rows = cur.fetchall()
    conn.close()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P2_MW", "P3_MW", "Status", "P1_MW", "||H||", "MaxEqMismatch(pu)", "Steps", "Time(s)", "Note", "Timestamp"])
        for r in rows:
            w.writerow(r)
    print(f"CSV 已导出: {csv_path} (rows={len(rows)})")


def plot_feasible_from_db(db_path: str, title: str = "TJU-9bus 可行域 (颜色为P1)", cmap: str = "viridis"):
    import matplotlib.pyplot as plt
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT p2, p3, p1 FROM scan_results WHERE status='RSEP';")
    rows = cur.fetchall()
    conn.close()
    if len(rows) == 0:
        print("没有可行点可绘制。")
        return
    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    c = [r[2] for r in rows]
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=c, cmap=cmap, s=30, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, label="P1 (MW)")
    plt.xlabel("P2 (MW)")
    plt.ylabel("P3 (MW)")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ======================== P2/P3 扫描到数据库 ========================

@dataclass
class ScanOptions:
    load_factor: float = 1.0
    verbose_each: bool = False
    commit_batch: int = 50  # 批量提交，减少IO
    show_progress_every: int = 10  # 进度打印频率（点）


def scan_p2_p3_to_db(p2_values: np.ndarray,
                     p3_values: np.ndarray,
                     db_path: str,
                     tju_opts: TJUOptions = TJUOptions(),
                     scan_opts: ScanOptions = ScanOptions()):
    total = len(p2_values) * len(p3_values)
    print(f"开始扫描 P2/P3 网格: {len(p2_values)} × {len(p3_values)} = {total} 点")
    print(f"数据库: {db_path}")

    conn = init_database(db_path, reset=True)
    cur = conn.cursor()

    t0 = time.time()
    n_done = 0
    to_commit = 0

    for i, p2 in enumerate(p2_values):
        for j, p3 in enumerate(p3_values):
            ret = tju_solve(load_factor=scan_opts.load_factor,
                            opts=tju_opts,
                            pv_pg_MW={2: float(p2), 3: float(p3)})
            insert_result(cur, float(p2), float(p3), ret)
            n_done += 1
            to_commit += 1

            if scan_opts.verbose_each:
                print(f"[{n_done:4d}/{total}] P2={p2:.1f}MW, P3={p3:.1f}MW -> {ret.status}, "
                      f"P1={(ret.Pg1 if ret.status=='RSEP' else np.nan):.1f}MW, ||H||={ret.res_norm:.2e}, steps={ret.steps}, {ret.message}")

            # 批量提交
            if to_commit >= scan_opts.commit_batch:
                conn.commit()
                to_commit = 0

            # 进度打印
            if (n_done % scan_opts.show_progress_every == 0) or (n_done == total):
                elapsed = time.time() - t0
                print(f"进度: {n_done}/{total} ({100*n_done/total:.1f}%) | 耗时: {elapsed:.1f}s | 平均: {elapsed/max(1,n_done):.3f}s/点")

    # 最后提交并关闭
    if to_commit > 0:
        conn.commit()

    # 汇总统计（SQL聚合，避免加载到内存）
    cur.execute("SELECT COUNT(*) FROM scan_results;")
    n_all = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM scan_results WHERE status='RSEP';")
    n_feas = cur.fetchone()[0]
    cur.execute("SELECT AVG(time), AVG(res_norm) FROM scan_results;")
    avg_time, avg_res = cur.fetchone()

    conn.close()

    elapsed = time.time() - t0
    print("\n=== 扫描完成 ===")
    print(f"总点数: {n_all}, 可行: {n_feas}, 成功率: {100.0*n_feas/max(1,n_all):.1f}%")
    print(f"总耗时: {elapsed:.2f}s, 平均时间/点: {elapsed/max(1,n_all):.3f}s")
    print(f"平均 ||H||: {avg_res if avg_res is not None else float('nan'):.3e}")


# ======================== 演示入口 ========================

if __name__ == "__main__":
    # 提升 matmul 精度/性能（可选）
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # 参数设置
    load_factor = 1.0
    tju_opts = TJUOptions(
        max_steps=20000,
        tol_res=1e-6,      # 放宽容差有助提速；需要更精密再调回
        dt_init=0.05,
        dt_inc=1.05,
        dt_dec=0.5,
        dt_max=1.0,
        verbose=False,       # 单点调试可开 True 查看迭代日志
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="auto",        # cuda->float32, cpu->float64
        seed=2025,
        print_every=100,
        max_time=120        # 单点最多3秒
    )
    scan_opts = ScanOptions(
        load_factor=load_factor,
        verbose_each=True,       # 打印每个点的简要结果
        commit_batch=50,
        show_progress_every=10
    )

    # P2 / P3 扫描范围（MW）
    n_points = 10
    p2_values = np.linspace(10, 180, n_points)
    p3_values = np.linspace(10, 180, n_points)

    # 数据库路径
    db_path = "tju_9bus_p2p3_scan.db"

    # 扫描并落库（内存占用极小）
    scan_p2_p3_to_db(p2_values, p3_values, db_path, tju_opts, scan_opts)

    # 可选：导出 CSV
    try:
        export_csv_from_db(db_path, "tju_9bus_p2p3_scan.csv")
    except Exception as e:
        print(f"导出CSV失败（可忽略）: {e}")

    # 可选：从数据库绘制可行域（需要 matplotlib）
    try:
        plot_feasible_from_db(db_path, title=f"TJU-9bus 可行域 (load_factor={load_factor})")
    except Exception as e:
        print(f"绘图失败（可忽略）: {e}")