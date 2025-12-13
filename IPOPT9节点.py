# filename: case9_feasible_region_tju_lines_fixedgrid_db.py
import os
import math
import time
import sqlite3
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
import math as _math

# ===========================
# TJU_v4 优化器（按你提供版本）
# ===========================
class TJU_v4(Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            beta_h=0.85,
            eps=1e-8,
            rebound='constant',
            warmup=100,
            init_lr=None,
            weight_decay=0.0,
            weight_decay_type='L2',
            hessian_scale=0.05,
            total_steps=10000,
            use_cosine_scheduler=True
    ):
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if weight_decay_type not in ['L2', 'stable', 'AdamW']:
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type}")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta_h=beta_h,
            eps=eps,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr or lr / 1000.0,
            base_lr=lr,
            weight_decay=weight_decay,
            weight_decay_type=weight_decay_type,
            hessian_scale=hessian_scale,
            total_steps=total_steps,
            use_cosine_scheduler=use_cosine_scheduler
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TJU_AdamW_Fixed不支持稀疏梯度")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                current_lr = self._compute_lr(group, step)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                if group['weight_decay_type'] == 'L2' and group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                step_size = current_lr / bias_corr1

                delta_grad = grad - (exp_avg / bias_corr1)
                approx_hessian.mul_(group['beta_h']).addcmul_(delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)
                else:
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(),
                                              torch.tensor(bound_val, device=p.device))

                denom = (exp_avg_sq.sqrt() / _math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian,
                    alpha=1.0
                ).add_(group['eps'])

                update = exp_avg / denom

                if group['weight_decay_type'] == 'stable' and group['weight_decay'] != 0:
                    decay_factor = group['weight_decay'] / denom.mean().clamp(min=1e-8)
                    update.add_(p, alpha=decay_factor)

                if group['weight_decay_type'] == 'AdamW' and group['weight_decay'] != 0:
                    p.data.mul_(1 - group['weight_decay'] * current_lr)

                p.add_(update, alpha=-step_size)
        return loss

    def _compute_lr(self, group, step):
        if step <= group['warmup']:
            return group['init_lr'] + (group['base_lr'] - group['init_lr']) * step / group['warmup']
        if not group['use_cosine_scheduler']:
            return group['base_lr']
        t = step - group['warmup']
        T = group['total_steps'] - group['warmup']
        if t <= T:
            return group['base_lr'] * (0.5 * (1 + _math.cos(_math.pi * t / T)))
        return group['base_lr'] * 0.01

# ===========================
# 9-bus 数据（case9mod 变体）
# ===========================
baseMVA = 100.0

bus = np.array([
    [1, 3, 0,   0,   0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
    [2, 2, 0,   0,   0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
    [3, 2, 0,   0,   0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
    [4, 1, 0,   0,   0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
    [5, 1, 54,  18,  0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
    [6, 1, 0,   0,   0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
    [7, 1, 60,  21,  0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
    [8, 1, 0,   0,   0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
    [9, 1, 75,  30,  0, 0, 1, 1.0, 0, 345, 1, 1.1, 0.9],
], dtype=float)

gen = np.array([
    [1,   0,  0, 300,  -5, 1, 100, 1, 250, 10, 0,0,0,0,0,0,0,0,0,0,0],
    [2, 163,  0, 300,  -5, 1, 100, 1, 300, 10, 0,0,0,0,0,0,0,0,0,0,0],
    [3,  85,  0, 300,  -5, 1, 100, 1, 270, 10, 0,0,0,0,0,0,0,0,0,0,0],
], dtype=float)

branch = np.array([
    [1, 4, 0.0000, 0.0576, 0.000, 250, 250, 250, 0, 0, 1, -360, 360],
    [4, 5, 0.0170, 0.0920, 0.158, 250, 250, 250, 0, 0, 1, -360, 360],
    [5, 6, 0.0390, 0.1700, 0.358, 150, 150, 150, 0, 0, 1, -360, 360],
    [3, 6, 0.0000, 0.0586, 0.000, 300, 300, 300, 0, 0, 1, -360, 360],
    [6, 7, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
    [7, 8, 0.0085, 0.0720, 0.149, 250, 250, 250, 0, 0, 1, -360, 360],
    [8, 2, 0.0000, 0.0625, 0.000, 250, 250, 250, 0, 0, 1, -360, 360],
    [8, 9, 0.0320, 0.1610, 0.306, 250, 250, 250, 0, 0, 1, -360, 360],
    [9, 4, 0.0100, 0.0850, 0.176, 250, 250, 250, 0, 0, 1, -360, 360],
], dtype=float)

# ===========================
# 预处理（张量/导纳/线路参数）
# ===========================
device = torch.device('cpu')
dtype = torch.double

n_bus = bus.shape[0]
n_gen = gen.shape[0]
n_branch = branch.shape[0]

Pd_pu = torch.tensor(bus[:,2]/baseMVA, dtype=dtype, device=device)
Qd_pu = torch.tensor(bus[:,3]/baseMVA, dtype=dtype, device=device)
Vmax = torch.tensor(bus[:,11], dtype=dtype, device=device)
Vmin = torch.tensor(bus[:,12], dtype=dtype, device=device)

gen_pmax = torch.tensor(gen[:,8]/baseMVA, dtype=dtype, device=device)
gen_pmin = torch.tensor(gen[:,9]/baseMVA, dtype=dtype, device=device)
gen_qmax = torch.tensor(gen[:,3]/baseMVA, dtype=dtype, device=device)
gen_qmin = torch.tensor(gen[:,4]/baseMVA, dtype=dtype, device=device)

# 生成器-母线映射
gen_buses = [int(g[0])-1 for g in gen]
bus_has_gen = [False]*n_bus
gen_map = [-1]*n_bus
for i, bidx in enumerate(gen_buses):
    bus_has_gen[bidx] = True
    gen_map[bidx] = i

# Ybus
Ybus = np.zeros((n_bus, n_bus), dtype=complex)
for br in branch:
    f = int(br[0])-1
    t = int(br[1])-1
    r, x, b = br[2], br[3], br[4]
    if abs(r) < 1e-12 and abs(x) > 0:
        y = 1/complex(0.0, x)
    else:
        z = complex(r, x)
        y = 1/z
    b_sh = complex(0.0, b/2.0)
    Ybus[f,f] += y + b_sh
    Ybus[t,t] += y + b_sh
    Ybus[f,t] -= y
    Ybus[t,f] -= y

G = torch.tensor(Ybus.real, dtype=dtype, device=device)
B = torch.tensor(Ybus.imag, dtype=dtype, device=device)

# 线路参数（用于 |S_ft|、|S_tf|）
branch_f = torch.tensor(branch[:,0]-1, dtype=torch.long, device=device)
branch_t = torch.tensor(branch[:,1]-1, dtype=torch.long, device=device)
r = branch[:,2]
x = branch[:,3]
b_total = branch[:,4]
ys = 1/(r + 1j*x)
g_series = np.real(ys)
b_series = np.imag(ys)

branch_g_series = torch.tensor(g_series, dtype=dtype, device=device)
branch_b_series = torch.tensor(b_series, dtype=dtype, device=device)
branch_b_shunt = torch.tensor(b_total, dtype=dtype, device=device)
branch_rateA_pu = torch.tensor(branch[:,5]/baseMVA, dtype=dtype, device=device)

# ===========================
# DB
# ===========================
DB_FILE = "case9_feasible_region_tju_lines_fixedgrid.db"

def init_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)""")
    c.execute("""CREATE TABLE grid_points (
        P2_MW REAL, P3_MW REAL,
        status TEXT, message TEXT, elapsed_s REAL
    )""")
    c.execute("""CREATE TABLE feasible_points (
        P1_MW REAL, P2_MW REAL, P3_MW REAL
    )""")
    conn.commit()
    return conn

# ===========================
# 工具：二次罚项
# ===========================
def quad_violation(x, lo, hi):
    below = (lo - x).clamp(min=0.0)
    above = (x - hi).clamp(min=0.0)
    return below**2 + above**2

# ===========================
# 线路两端视在功率
# ===========================
def line_S_ends(Vm_full, Va_full):
    f = branch_f
    t = branch_t
    g = branch_g_series
    b = branch_b_series
    bc = branch_b_shunt

    Vf = Vm_full[f]
    Vt = Vm_full[t]
    theta_ft = Va_full[f] - Va_full[t]
    cos_ft = torch.cos(theta_ft)
    sin_ft = torch.sin(theta_ft)

    P_ft = g * Vf*Vf - g*Vf*Vt*cos_ft - b*Vf*Vt*sin_ft
    Q_ft = -(b + bc/2.0) * Vf*Vf + b*Vf*Vt*cos_ft - g*Vf*Vt*sin_ft
    S_ft = torch.sqrt(P_ft*P_ft + Q_ft*Q_ft + 1e-12)

    P_tf = g * Vt*Vt - g*Vf*Vt*cos_ft + b*Vf*Vt*sin_ft
    Q_tf = -(b + bc/2.0) * Vt*Vt + b*Vf*Vt*cos_ft + g*Vf*Vt*sin_ft
    S_tf = torch.sqrt(P_tf*P_tf + Q_tf*Q_tf + 1e-12)
    return S_ft, S_tf

# ===========================
# 损失函数（包含全部约束）
# ===========================
def build_loss(Pg1_pu, Qg1_pu, Qg2_pu, Qg3_pu, Va_vars, Vm_vars, p2_pu, p3_pu,
               penalty_scale=10.0, penalty_scale_line=20.0):
    # Va1=0, Vm1=Vm2=Vm3=1.0
    Va_full = torch.cat([torch.zeros(1, dtype=dtype, device=device), Va_vars])   # [1] + [Va2..Va9]
    Vm_full = torch.cat([torch.ones(3, dtype=dtype, device=device), Vm_vars])    # [Vm1..Vm3]=1, + [Vm4..Vm9]

    Va_i = Va_full.unsqueeze(1)
    Va_j = Va_full.unsqueeze(0)
    theta = Va_i - Va_j

    Vm_i = Vm_full.unsqueeze(1)
    Vm_j = Vm_full.unsqueeze(0)

    cosT = torch.cos(theta)
    sinT = torch.sin(theta)

    P_terms = Vm_i * (G * cosT + B * sinT) * Vm_j
    Q_terms = Vm_i * (G * sinT - B * cosT) * Vm_j

    P_calc = P_terms.sum(dim=1)
    Q_calc = Q_terms.sum(dim=1)

    # 调度注入
    P_sch = torch.zeros(n_bus, dtype=dtype, device=device)
    Q_sch = torch.zeros(n_bus, dtype=dtype, device=device)
    P_sch[0] = Pg1_pu - Pd_pu[0];  Q_sch[0] = Qg1_pu - Qd_pu[0]
    P_sch[1] = p2_pu   - Pd_pu[1]; Q_sch[1] = Qg2_pu - Qd_pu[1]
    P_sch[2] = p3_pu   - Pd_pu[2]; Q_sch[2] = Qg3_pu - Qd_pu[2]
    for k in range(3, n_bus):
        P_sch[k] = -Pd_pu[k]
        Q_sch[k] = -Qd_pu[k]

    # 平衡残差
    loss_p = ((P_calc - P_sch) ** 2).sum()
    loss_q = ((Q_calc - Q_sch) ** 2).sum()

    # 发电机/电压边界
    loss_pg1 = quad_violation(Pg1_pu, gen_pmin[0], gen_pmax[0])
    loss_qg = quad_violation(Qg1_pu, gen_qmin[0], gen_qmax[0])
    loss_qg += quad_violation(Qg2_pu, gen_qmin[1], gen_qmax[1])
    loss_qg += quad_violation(Qg3_pu, gen_qmin[2], gen_qmax[2])

    vm_pen = torch.zeros(1, dtype=dtype, device=device)
    for idx in range(3, n_bus):  # PQ buses 4..9
        vm_pen = vm_pen + quad_violation(Vm_full[idx], Vmin[idx], Vmax[idx])

    # 线路限值
    S_ft, S_tf = line_S_ends(Vm_full, Va_full)
    loss_line = ((S_ft - branch_rateA_pu).clamp(min=0.0) ** 2).sum()
    loss_line += ((S_tf - branch_rateA_pu).clamp(min=0.0) ** 2).sum()

    loss_constraints = loss_pg1 + loss_qg + vm_pen
    loss_total = loss_p + loss_q + penalty_scale * loss_constraints + penalty_scale_line * loss_line
    return loss_total, loss_p, loss_q, loss_constraints, loss_line

# ===========================
# 单点求解（TJU_v4）
# ===========================
def solve_with_tju(p2_MW, p3_MW,
                   max_steps=3000, tol=1e-5,
                   penalty_scale=10.0, penalty_scale_line=20.0,
                   lr=0.01, weight_decay=1e-4, use_cosine=True):
    p2_pu = p2_MW / baseMVA
    p3_pu = p3_MW / baseMVA

    total_load_pu = Pd_pu.sum().item()
    pg1_init = float(np.clip(total_load_pu - p2_pu - p3_pu,
                             gen_pmin[0].item(), gen_pmax[0].item()))

    Pg1_pu = torch.tensor(pg1_init, dtype=dtype, device=device, requires_grad=True)
    Qg1_pu = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    Qg2_pu = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    Qg3_pu = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    Va_vars = torch.zeros(8, dtype=dtype, device=device, requires_grad=True)  # Va2..Va9
    Vm_vars = torch.ones(6, dtype=dtype, device=device, requires_grad=True)   # Vm4..Vm9

    params = [Pg1_pu, Qg1_pu, Qg2_pu, Qg3_pu, Va_vars, Vm_vars]
    opt = TJU_v4(params,
                 lr=lr,
                 weight_decay=weight_decay,
                 weight_decay_type='AdamW',
                 total_steps=max_steps,
                 use_cosine_scheduler=use_cosine,
                 warmup=max(50, int(0.02*max_steps)),
                 hessian_scale=0.05,
                 beta_h=0.85)

    best_val = float('inf')
    best_state = None

    for step in range(1, max_steps+1):
        opt.zero_grad()
        total_loss, lp, lq, lc, ll = build_loss(
            Pg1_pu, Qg1_pu, Qg2_pu, Qg3_pu, Va_vars, Vm_vars,
            torch.tensor(p2_pu, dtype=dtype, device=device),
            torch.tensor(p3_pu, dtype=dtype, device=device),
            penalty_scale=penalty_scale,
            penalty_scale_line=penalty_scale_line
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        val = total_loss.item()
        if val < best_val:
            best_val = val
            best_state = (
                Pg1_pu.detach().clone(), Qg1_pu.detach().clone(),
                Qg2_pu.detach().clone(), Qg3_pu.detach().clone(),
                Va_vars.detach().clone(), Vm_vars.detach().clone()
            )
        if val < tol:
            break

    if best_state is None:
        return False, None

    Pg1_b, Qg1_b, Qg2_b, Qg3_b, Va_b, Vm_b = best_state
    with torch.no_grad():
        total_loss_b, lp2, lq2, lc2, ll2 = build_loss(
            Pg1_b, Qg1_b, Qg2_b, Qg3_b, Va_b, Vm_b,
            torch.tensor(p2_pu, dtype=dtype, device=device),
            torch.tensor(p3_pu, dtype=dtype, device=device),
            penalty_scale=penalty_scale,
            penalty_scale_line=penalty_scale_line
        )
        feasible = (total_loss_b.item() < max(tol, 1e-6)) \
                   and (lc2.item() < 1e-6) \
                   and (ll2.item() < 1e-6) \
                   and ((lp2.item() + lq2.item()) < 1e-4)

    if feasible:
        return True, Pg1_b.item() * baseMVA
    else:
        return False, None

# ===========================
# 固定网格扫描（带进度条与打印）
# ===========================
def scan_fixed_grid_and_visualize(n_points=100):
    """
    - 单阶段固定网格：n_points x n_points
    - tqdm 进度条 + 每 1% 进度打印摘要
    - 每点打印结果行
    """
    conn = init_db()
    c = conn.cursor()

    # 扫描边界
    p2_min, p2_max = gen_pmin[1].item()*baseMVA, gen_pmax[1].item()*baseMVA
    p3_min, p3_max = gen_pmin[2].item()*baseMVA, gen_pmax[2].item()*baseMVA

    # 固定网格
    p2_values = np.linspace(10, 180, n_points)
    p3_values = np.linspace(10, 180, n_points)

    total = n_points * n_points
    c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", ("grid", f"{n_points}x{n_points}"))
    c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", ("p2_range", f"{p2_min},{p2_max}"))
    c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", ("p3_range", f"{p3_min},{p3_max}"))
    conn.commit()

    print(f"[Scan] 固定网格 {n_points}x{n_points} = {total} 个点，开始计算 ...")
    feasible_count = 0
    start_time = time.time()
    next_percent_print = 1  # 每 1% 打印一次摘要

    # tqdm 进度条
    pbar = tqdm(total=total, ncols=100, unit="pt")

    done = 0
    for i, p2 in enumerate(p2_values):
        for j, p3 in enumerate(p3_values):
            t0 = time.time()
            ok, p1_MW = solve_with_tju(
                p2, p3,
                max_steps=3000, tol=5e-6,
                penalty_scale=10.0, penalty_scale_line=20.0,
                lr=0.01, weight_decay=1e-4, use_cosine=True
            )
            dt = time.time() - t0

            if ok:
                feasible_count += 1
                c.execute("INSERT INTO feasible_points(P1_MW,P2_MW,P3_MW) VALUES(?,?,?)", (p1_MW, p2, p3))
                c.execute("INSERT INTO grid_points(P2_MW,P3_MW,status,message,elapsed_s) VALUES(?,?,?,?,?)",
                          (p2, p3, "solved", "", dt))
                print(f"[OK ] P2={p2:.2f} MW, P3={p3:.2f} MW -> P1={p1_MW:.2f} MW, time={dt:.2f}s")
            else:
                c.execute("INSERT INTO grid_points(P2_MW,P3_MW,status,message,elapsed_s) VALUES(?,?,?,?,?)",
                          (p2, p3, "infeasible", "", dt))
                print(f"[FAIL] P2={p2:.2f} MW, P3={p3:.2f} MW -> infeasible, time={dt:.2f}s")

            conn.commit()

            done += 1
            pbar.update(1)

            # 每 1% 打印一次摘要
            progress_pct = done * 100.0 / total
            if progress_pct >= next_percent_print - 1e-9:
                elapsed = time.time() - start_time
                rate = done / max(elapsed, 1e-9)
                remain = (total - done) / max(rate, 1e-9)
                print(f"== Progress {progress_pct:.0f}% | {done}/{total} | "
                      f"elapsed {elapsed/60:.1f} min | ETA {remain/60:.1f} min | feasible={feasible_count}")
                next_percent_print += 1

    pbar.close()
    conn.close()
    print(f"扫描结束：可行点 {feasible_count}/{total}，数据库：{DB_FILE}")

    # 可视化
    visualize_p1_p2()

# ===========================
# 可视化函数
# ===========================
def visualize_p1_p2():
    if not os.path.exists(DB_FILE):
        print("数据库不存在，请先扫描。")
        return
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT P1_MW,P2_MW,P3_MW FROM feasible_points")
    rows = c.fetchall()
    conn.close()

    if len(rows) == 0:
        print("无可行点。")
        return

    p1 = [r[0] for r in rows]
    p2 = [r[1] for r in rows]
    p3 = [r[2] for r in rows]

    plt.figure(figsize=(10,7))
    sc = plt.scatter(p1, p2, c=p3, cmap="viridis", s=12, edgecolors='k', linewidths=0.2, alpha=0.9)
    cbar = plt.colorbar(sc)
    cbar.set_label("P3 (MW)")
    plt.xlabel("P1 (MW)")
    plt.ylabel("P2 (MW)")
    plt.title(f"9-bus 可行域P1–P2，颜色=P3；可行点={len(p1)}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===========================
# 入口
# ===========================
if __name__ == "__main__":
    # 按需调整 n_points（如算力不足可先用 30~50）
    scan_fixed_grid_and_visualize(n_points=10)