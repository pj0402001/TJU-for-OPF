import torch
import numpy as np
import sqlite3
import math
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import matplotlib
from torch.optim import Optimizer

# Pyomo
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# 优先使用 Appsi-Ipopt（cyipopt），不可用时回退
try:
    from pyomo.contrib.appsi.solvers import Ipopt as AppsiIpopt
except Exception:
    AppsiIpopt = None

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# ===== 系统参数（2 节点） =====
mpc = {
    "baseMVA": 100,
    "bus": [
        [1, 3, 0,   0,   0, 0, 1, 0.964, 0,   0, 1, 1.05, 0.95],
        [2, 1, 350, -350, 0, 0, 1, 1.0,   -65, 0, 1, 1.05, 0.95]
    ],
    "gen": [
        [1, 400, 100, 400, -400, 0.964, 100, 1, 600, 0]
    ],
    "branch": [
        [1, 2, 0.04, 0.2, 0, 990000, 0, 0, 0, 0, 1, -360, 360]
    ]
}

# ================= 数据库 (修改：增加 Qg1_Mvar 字段) =================
def init_db():
    if os.path.exists('opf_results_tju_improved.db'):
        os.remove('opf_results_tju_improved.db')
    conn = sqlite3.connect('opf_results_tju_improved.db')
    cursor = conn.cursor()
    # 修改：增加了 Qg1_Mvar 字段
    cursor.execute('''CREATE TABLE feasible_points
                    (Pg1_MW REAL, V1 REAL, V2 REAL, Qg1_Mvar REAL, loss REAL, epoch_count INTEGER)''')
    cursor.execute('''CREATE TABLE partial_results
                    (Pg1_MW REAL, best_loss REAL, epoch_count INTEGER)''')
    conn.commit()
    return conn

# ============== 初始化（根据 Pg 轻微自适应） ==============
def initialize_variables(Pg1_pu):
    base_V1 = 0.965 - 0.003 * (Pg1_pu - 4.4)
    base_V2 = 1.02  - 0.015 * (Pg1_pu - 4.4)
    return (
        torch.tensor([base_V1], requires_grad=True),
        torch.tensor([base_V2], requires_grad=True),
        torch.tensor([0.01], requires_grad=True),
        torch.tensor([0.0],  requires_grad=True)
    )

# ============== 平滑边界罚：softplus ==============
def smooth_barrier(x, lower, upper, coef=0.05, delta=0.02):
    sp = torch.nn.functional.softplus
    lower_viol = sp((lower - x) / delta) * delta
    upper_viol = sp((x - upper) / delta) * delta
    return coef * torch.sum(lower_viol + upper_viol)

# ============== 2 节点潮流方程（PyTorch 张量） ==============
def power_flow_eq(V1, V2, theta2, Pg1_pu, Qg1, P_load_pu, Q_load_pu, R_pu, X_pu):
    Y = 1 / complex(R_pu, X_pu)
    G_pu, B_line_pu = Y.real, Y.imag

    theta12 = -theta2
    cos12 = torch.cos(theta12)
    sin12 = torch.sin(theta12)
    V1V2 = V1 * V2

    P_inj1 = G_pu * V1**2 - V1V2 * (G_pu * cos12 + B_line_pu * sin12)
    Q_inj1 = -B_line_pu * V1**2 - V1V2 * (G_pu * sin12 - B_line_pu * cos12)
    P_inj2 = G_pu * V2**2 - V1V2 * (G_pu * cos12 - B_line_pu * sin12)
    Q_inj2 = -B_line_pu * V2**2 + V1V2 * (G_pu * sin12 + B_line_pu * cos12)

    P_res1 = P_inj1 - Pg1_pu
    Q_res1 = Q_inj1 - Qg1
    P_res2 = P_inj2 + P_load_pu
    Q_res2 = Q_inj2 + Q_load_pu
    return P_res1, Q_res1, P_res2, Q_res2

# ============== 改进的 TJU 优化器 ==============
class TJU_Improved(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        betas=(0.9, 0.999),
        beta_h=0.85,
        eps=1e-8,
        rebound='constant',
        warmup=100,
        init_lr=None,
        weight_decay=0,
        weight_decay_type='stable',
        hessian_scale=0.05,
        total_steps=5000,
        use_cosine_scheduler=False
    ):
        defaults = dict(lr=lr, betas=betas, beta_h=beta_h, eps=eps,
                        rebound=rebound, warmup=warmup, init_lr=init_lr or lr/1000.0,
                        base_lr=lr, weight_decay=weight_decay, weight_decay_type=weight_decay_type,
                        hessian_scale=hessian_scale, total_steps=total_steps,
                        use_cosine_scheduler=use_cosine_scheduler)
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
                    raise RuntimeError("不支持稀疏梯度")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                current_lr = group['lr']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                step_size = current_lr / bias_corr1

                delta_grad = grad - (exp_avg / bias_corr1)
                approx_hessian.mul_(group['beta_h']).addcmul_(
                    delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)
                else:
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(),
                                              torch.tensor(bound_val, device=p.device))

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian,
                    alpha=1.0
                ).add_(group['eps'])

                update = exp_avg / denom
                p.add_(update, alpha=-step_size)
        return loss

# ============== 梯度范数 ==============
def grad_norm(params):
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float((p.grad.detach()**2).sum().item())
    return math.sqrt(sq)

# ============== 优先 Appsi-Ipopt 的 Pyomo 求解（含初值钳制） ==============
def solve_with_ipopt_2bus(init, Pg1_pu, P_load_pu, Q_load_pu, R_pu, X_pu,
                          Vmin, Vmax, Qg_min, Qg_max,
                          ipopt_max_iter=300, ipopt_tol=1e-8, silent=True):
    """
    init: dict {'V1','V2','theta2','Qg1'}
    返回: (ok, sol_dict, solver_results)
    """
    try:
        # ====== 初值钳制，避免 W1002 并提高内点法稳健性 ======
        V1_init = float(np.clip(init['V1'], Vmin + 1e-4, Vmax - 1e-4))
        V2_init = float(np.clip(init['V2'], Vmin + 1e-4, Vmax - 1e-4))
        theta2_init = float(np.clip(init['theta2'], -math.pi/2 + 1e-4, math.pi/2 - 1e-4))
        Qg1_init = float(np.clip(init['Qg1'], Qg_min + 1e-4, Qg_max - 1e-4))

        # ====== 建模 ======
        model = pyo.ConcreteModel()
        model.V1 = pyo.Var(bounds=(Vmin, Vmax), initialize=V1_init)
        model.V2 = pyo.Var(bounds=(Vmin, Vmax), initialize=V2_init)
        model.theta2 = pyo.Var(bounds=(-math.pi/2, math.pi/2), initialize=theta2_init)
        model.Qg1 = pyo.Var(bounds=(Qg_min, Qg_max), initialize=Qg1_init)

        # 常量
        Y = 1 / complex(R_pu, X_pu)
        G = float(Y.real)
        B = float(Y.imag)

        def p_res1(m):
            return G*m.V1**2 - m.V1*m.V2*(G*pyo.cos(-m.theta2) + B*pyo.sin(-m.theta2)) - Pg1_pu == 0
        def q_res1(m):
            return -B*m.V1**2 - m.V1*m.V2*(G*pyo.sin(-m.theta2) - B*pyo.cos(-m.theta2)) - m.Qg1 == 0
        def p_res2(m):
            return G*m.V2**2 - m.V1*m.V2*(G*pyo.cos(-m.theta2) - B*pyo.sin(-m.theta2)) + P_load_pu == 0
        def q_res2(m):
            return -B*m.V2**2 + m.V1*m.V2*(G*pyo.sin(-m.theta2) + B*pyo.cos(-m.theta2)) + Q_load_pu == 0

        model.c1 = pyo.Constraint(rule=p_res1)
        model.c2 = pyo.Constraint(rule=q_res1)
        model.c3 = pyo.Constraint(rule=p_res2)
        model.c4 = pyo.Constraint(rule=q_res2)

        # 可行性问题：目标置零
        model.obj = pyo.Objective(expr=0.0, sense=pyo.minimize)

        # ====== 1) 优先使用 Appsi-Ipopt（cyipopt） ======
        try:
            if AppsiIpopt is None:
                raise RuntimeError("Appsi-Ipopt 不可用或导入失败")
            solver = AppsiIpopt()
            solver.options.update({
                "print_level": 0 if silent else 5,
                "tol": ipopt_tol,
                "acceptable_tol": max(ipopt_tol*10, 1e-6),
                "max_iter": ipopt_max_iter,
                "mu_strategy": "adaptive",
                "linear_solver": "mumps",
                "hessian_approximation": "limited-memory",
            })
            res = solver.solve(model, tee=not silent)
            term = getattr(res, "termination_condition", None)
            ok = term in (pyo.TerminationCondition.optimal,
                          pyo.TerminationCondition.locallyOptimal)
            if ok:
                sol = {
                    'V1': pyo.value(model.V1),
                    'V2': pyo.value(model.V2),
                    'theta2': pyo.value(model.theta2),
                    'Qg1': pyo.value(model.Qg1),
                }
                return True, sol, res
            # 未最优则回退外部 ipopt
            if not silent:
                print("[Appsi-Ipopt] 未达到最优，回退到外部 ipopt ...")
        except Exception as e_appsi:
            if not silent:
                print(f"[Appsi-Ipopt异常] {e_appsi}，回退外部 ipopt。")

        # ====== 2) 回退：外部 ipopt 可执行文件（可能受 MKL 影响） ======
        solver = SolverFactory('ipopt')
        if (solver is None) or (not solver.available(False)):
            if not silent:
                print("[外部Ipopt] 不可用（未安装或不在 PATH）")
            return False, None, None

        solver.options['print_level'] = 0 if silent else 5
        solver.options['tol'] = ipopt_tol
        solver.options['acceptable_tol'] = max(ipopt_tol*10, 1e-6)
        solver.options['max_iter'] = ipopt_max_iter
        solver.options['mu_strategy'] = 'adaptive'
        solver.options['linear_solver'] = 'mumps'
        solver.options['hessian_approximation'] = 'limited-memory'

        res = solver.solve(model, tee=not silent)
        ok = (res.solver.termination_condition == pyo.TerminationCondition.optimal)
        if ok:
            sol = {
                'V1': pyo.value(model.V1),
                'V2': pyo.value(model.V2),
                'theta2': pyo.value(model.theta2),
                'Qg1': pyo.value(model.Qg1),
            }
            return True, sol, res
        else:
            return False, None, res

    except Exception as e:
        print(f"[IPOPT异常] {e}")
        return False, None, None

# ============== 主流程：TJU + IPOPT 双阶段 ==============
def compute_feasible_region_hybrid():
    conn = init_db()
    S_base = mpc["baseMVA"]
    P_load_pu = mpc["bus"][1][2] / S_base
    Q_load_pu = mpc["bus"][1][3] / S_base
    R_pu = mpc["branch"][0][2]
    X_pu = mpc["branch"][0][3]

    Vmin, Vmax = 0.95, 1.05
    Qg_min, Qg_max = -4.0, 4.0

    num_points = 200
    Pg1_points_MW = np.linspace(440, 461.5, num_points)

    with open("optimization_log.csv", "w") as log_file:
        log_file.write("Pg1_MW,Start_V1,Start_V2,Best_V1,Best_V2,Final_Loss,Epochs,Switched,IpoptOk\n")

        for Pg1_MW in tqdm(Pg1_points_MW, desc="TJU+IPOPT双阶段计算"):
            Pg1_pu = Pg1_MW / S_base
            V1, V2, theta2, Qg1 = initialize_variables(Pg1_pu)
            params = [V1, V2, theta2, Qg1]

            # 学习率略降以提升稳健性
            optimizer = TJU_Improved(params=params, lr=0.01, betas=(0.9, 0.999), beta_h=0.85)
            best_loss = float('inf')
            best_state = None
            start_time = time.time()

            # 阶段切换参数
            switch_loss_tol = 1e-1
            switch_grad_tol = 5e-1
            plateau_patience = 800
            min_change_counter = 0
            switched = False
            ipopt_ok = False

            # 动态 barrier 与权重
            base_barrier = 0.05

            max_epochs = 3000
            epoch_used = 0  # 初始化 epoch_used
            
            for epoch in range(max_epochs):
                epoch_used += 1  # 记录迭代次数

                # --- 前向 ---
                P_res1, Q_res1, P_res2, Q_res2 = power_flow_eq(
                    V1, V2, theta2, Pg1_pu, Qg1, P_load_pu, Q_load_pu, R_pu, X_pu
                )

                # 归一权重
                scale_p = 1.0 / max(abs(Pg1_pu) + abs(P_load_pu) + 1e-2, 1.0)
                scale_q = 1.0 / max(abs(Q_load_pu) + 1.0, 1.0)

                power_loss = scale_p*(P_res1**2 + P_res2**2) + scale_q*(Q_res1**2 + Q_res2**2)
                power_loss = 200.0 * power_loss  # 强化等式收敛

                # 动态 barrier（逐步收缩）
                barrier_coef = base_barrier * (0.5 ** (epoch // 2000))

                voltage_loss = smooth_barrier(V1, Vmin, Vmax, coef=barrier_coef) + \
                               smooth_barrier(V2, Vmin, Vmax, coef=barrier_coef)
                qg_loss = smooth_barrier(Qg1, Qg_min, Qg_max, coef=barrier_coef)
                angle_loss = smooth_barrier(theta2, -math.pi, math.pi, coef=barrier_coef)

                total_loss = power_loss + voltage_loss + qg_loss + angle_loss

                # --- 反向与步进 ---
                optimizer.zero_grad()
                total_loss.backward()
                # 角度梯度裁剪
                if theta2.grad is not None:
                    theta2.grad = torch.clamp(theta2.grad, -100, 100)
                optimizer.step()

                # 记录最好解
                prev_best = best_loss
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_state = (V1.item(), V2.item(), theta2.item(), Qg1.item())
                    min_change_counter = 0
                else:
                    min_change_counter += 1

                # 动态学习率衰减
                if epoch > 0 and epoch % 2000 == 0:
                    for g in optimizer.param_groups:
                        g['lr'] = max(g['lr'] * 0.8, 5e-6)

                # 每 1000 步打印一次
                if epoch % 1000 == 0:
                    gnorm = grad_norm(params)
                    print(f"Pg1={Pg1_MW:.1f}MW, Epoch={epoch}, Loss={total_loss.item():.6e}, gnorm={gnorm:.3e}")

                # ===== 切换判据 =====
                gnorm = grad_norm(params)
                if (best_loss <= switch_loss_tol) or (gnorm <= switch_grad_tol) or \
                   (min_change_counter > plateau_patience and abs(prev_best - best_loss) < 1e-6):
                    # 切换到 IPOPT 精修（优先 Appsi）
                    switched = True
                    init = {'V1': best_state[0], 'V2': best_state[1], 'theta2': best_state[2], 'Qg1': best_state[3]}

                    ok, sol, _res = solve_with_ipopt_2bus(
                        init=init, Pg1_pu=Pg1_pu, P_load_pu=P_load_pu, Q_load_pu=Q_load_pu,
                        R_pu=R_pu, X_pu=X_pu, Vmin=Vmin, Vmax=Vmax, Qg_min=Qg_min, Qg_max=Qg_max,
                        ipopt_max_iter=500, ipopt_tol=1e-8, silent=True
                    )
                    if ok and sol is not None:
                        # 用 IPOPT 结果更新
                        V1_val, V2_val = sol['V1'], sol['V2']
                        th_val, qg_val = float(sol['theta2']), float(sol['Qg1'])

                        # 评估最终残差损失（用于记录）
                        V1_t = torch.tensor([V1_val], requires_grad=False)
                        V2_t = torch.tensor([V2_val], requires_grad=False)
                        th_t = torch.tensor([th_val], requires_grad=False)
                        qg_t = torch.tensor([qg_val], requires_grad=False)
                        P_res1_t, Q_res1_t, P_res2_t, Q_res2_t = power_flow_eq(
                            V1_t, V2_t, th_t, Pg1_pu, qg_t, P_load_pu, Q_load_pu, R_pu, X_pu
                        )
                        final_loss = float(
                            200.0*(scale_p*(P_res1_t**2 + P_res2_t**2) + scale_q*(Q_res1_t**2 + Q_res2_t**2)).item()
                        )
                        best_loss = min(best_loss, final_loss)
                        best_state = (V1_val, V2_val, th_val, qg_val)
                        ipopt_ok = True

                    # 无论成功与否都结束该 Pg1 点的循环
                    break
            else:
                # 未触发切换则用最终 TJU 结果
                epoch_used = max_epochs

            # 检查约束并记录
            if best_state is not None:
                V1_val, V2_val, th_val, qg_val = best_state
                if not (Vmin <= V1_val <= Vmax) or not (Vmin <= V2_val <= Vmax) or not (Qg_min <= qg_val <= Qg_max):
                    print(f"[Warn] Pg1={Pg1_MW:.1f}MW 找到的结果不满足约束，舍弃")
                    continue  # 跳过当前循环

            end_time = time.time()
            compute_time = end_time - start_time
            start_V1, start_V2 = float(params[0].detach().item()), float(params[1].detach().item())

            # 记录日志
            log_line = f"{Pg1_MW},{start_V1:.4f},{start_V2:.4f},{best_state[0]:.4f},{best_state[1]:.4f},{best_loss:.6e},{epoch_used},{int(switched)},{int(ipopt_ok)}"
            log_file.write(log_line + "\n")

            # 存库（修改：保存 Qg1）
            # best_state 顺序: (V1, V2, theta2, Qg1)
            Qg1_Mvar = best_state[3] * S_base
            
            if best_loss < 0.01:
                conn.execute(
                    "INSERT INTO feasible_points (Pg1_MW, V1, V2, Qg1_Mvar, loss, epoch_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (Pg1_MW, best_state[0], best_state[1], Qg1_Mvar, best_loss, epoch_used)
                )
                print(f"[OK] Pg1={Pg1_MW:.1f}MW 可行: Qg1={Qg1_Mvar:.2f}Mvar, V1={best_state[0]:.4f}, V2={best_state[1]:.4f}, "
                      f"loss={best_loss:.3e}, iters={epoch_used}, switched={switched}, ipopt_ok={ipopt_ok}, time={compute_time:.2f}s")
            else:
                conn.execute(
                    "INSERT INTO partial_results (Pg1_MW, best_loss, epoch_count) VALUES (?, ?, ?)",
                    (Pg1_MW, best_loss, epoch_used)
                )
                print(f"[Warn] Pg1={Pg1_MW:.1f}MW 未可行: min_loss={best_loss:.3e}, iters={epoch_used}, switched={switched}, ipopt_ok={ipopt_ok}, time={compute_time:.2f}s")
            conn.commit()
    conn.close()

# ============== 可视化 (修改：生成 Pg1-Qg1 散点图) ==============
def visualize_results():
    if not os.path.exists('opf_results_tju_improved.db'):
        print("数据库未找到，请先运行计算过程")
        return
    conn = sqlite3.connect('opf_results_tju_improved.db')
    c = conn.cursor()
    try:
        # 修改：读取 Qg1_Mvar
        c.execute("SELECT Pg1_MW, V1, V2 FROM feasible_points")
        feasible_data = c.fetchall()
        
        c.execute("SELECT * FROM partial_results")
        partial_data = c.fetchall()
        conn.close()

        if not feasible_data and not partial_data:
            print("未找到任何结果！请查看日志文件分析原因")
            return

        plt.figure(figsize=(10, 6))

        # 绘制可行解 (Scatter Plot: V1 vs V2)
        # 修改后的代码部分：
        if feasible_data:
            data_arr = np.array(feasible_data)
            print(f"数据形状: {data_arr.shape}")  # 调试信息，显示数组维度
            
            V1 = data_arr[:, 1]  # 第一列是 Pg1_MW
            V2 = data_arr[:, 2]  # 第二列是 V1，第三列是 V2
            
            # 使用单一颜色绘制散点，展示V1-V2关系
            plt.scatter(V1, V2, c='blue', s=20, alpha=0.6, label="可行运行点 (TJU+IPOPT)")
            

        # 设置图表属性
        plt.xlabel("节点1电压 V1 (p.u.)", fontsize=12)
        plt.ylabel("节点2电压 V2 (p.u.)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.xlim(0.94, 0.96)
        plt.ylim(0.94, 1.05)
        # 保存并显示
        save_name = "hybrid_V1V2_scatter.png"
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"图表已保存为 {save_name}")
        plt.show()
        
    except sqlite3.OperationalError as e:
        print(f"数据库查询错误 (可能是数据库版本不匹配，请删除旧 .db 文件重试): {e}")
        if 'conn' in locals() and conn: conn.close()
    except Exception as e:
        print(f"可视化错误: {e}")
        if 'conn' in locals() and conn: conn.close()

# ============== 入口 ==============
if __name__ == "__main__":
    if os.path.exists('optimization_log.csv'):
        os.remove('optimization_log.csv')
    
    # 运行双阶段计算
    compute_feasible_region_hybrid()
    
    # 运行可视化
    visualize_results()
