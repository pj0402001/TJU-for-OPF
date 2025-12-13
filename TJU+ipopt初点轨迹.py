import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import math
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import sqlite3
import os

class TJU_v4(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_h=0.85, eps=1e-8,
                 rebound='constant', warmup=100, init_lr=None, weight_decay=0.0,
                 weight_decay_type='L2', hessian_scale=0.05, total_steps=10000,
                 use_cosine_scheduler=True):
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if weight_decay_type not in ['L2', 'stable', 'AdamW']:
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type}")

        defaults = dict(lr=lr, betas=betas, beta_h=beta_h, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr or lr / 1000.0, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type,
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
                    raise RuntimeError("TJU不支持稀疏梯度")

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
            return group['base_lr'] * (0.5 * (1 + math.cos(math.pi * t / T)))
        return group['base_lr'] * 0.01

# 系统参数（2节点网络）
mpc = {
    "baseMVA": 100,
    "bus": [
        [1, 3, 0, 0, 0, 0, 1, 0.964, 0, 0, 1, 1.05, 0.95],
        [2, 1, 350, -350, 0, 0, 1, 1.0, -65, 0, 1, 1.05, 0.95]
    ],
    "gen": [
        [1, 400, 100, 400, -400, 0.964, 100, 1, 600, 0]
    ],
    "branch": [
        [1, 2, 0.04, 0.2, 0, 990000, 0, 0, 0, 0, 1, -360, 360]
    ]
}

# 初始化变量
def initialize_variables(Pg1_pu):
    base_V1 = 0.965 - 0.003 * (Pg1_pu - 4.4)
    base_V2 = 1.02 - 0.015 * (Pg1_pu - 4.4)
    return (
        torch.tensor([base_V1], requires_grad=True),
        torch.tensor([base_V2], requires_grad=True),
        torch.tensor([0.01], requires_grad=True),
        torch.tensor([0.0], requires_grad=True)
    )

# 平滑边界罚
def smooth_barrier(x, lower, upper, coef=0.05, delta=0.02):
    sp = torch.nn.functional.softplus
    lower_viol = sp((lower - x) / delta) * delta
    upper_viol = sp((x - upper) / delta) * delta
    return coef * torch.sum(lower_viol + upper_viol)

# 2节点潮流方程
def power_flow_eq(V1, V2, theta2, Pg1_pu, Qg1, P_load_pu, Q_load_pu, R_pu, X_pu):
    Y = 1 / complex(R_pu, X_pu)
    G_pu, B_line_pu = Y.real, Y.imag

    theta12 = -theta2.item()  # Get the value as a scalar
    cos12 = torch.cos(torch.tensor(theta12))  # Ensure it is a tensor
    sin12 = torch.sin(torch.tensor(theta12))

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

# IPOPT求解函数
def solve_with_ipopt(init, Pg1_pu, P_load_pu, Q_load_pu, R_pu, X_pu,
                     Vmin, Vmax, Qg_min, Qg_max,
                     ipopt_max_iter=300, ipopt_tol=1e-8, silent=True):
    """
    initial values as a dictionary {'V1','V2','theta2','Qg1'}
    return: (ok, sol_dict, solver_results)
    """
    try:
        # Initialize constraints
        V1_init = float(np.clip(init['V1'], Vmin + 1e-4, Vmax - 1e-4))
        V2_init = float(np.clip(init['V2'], Vmin + 1e-4, Vmax - 1e-4))
        theta2_init = float(np.clip(init['theta2'], -math.pi / 2 + 1e-4, math.pi / 2 - 1e-4))
        Qg1_init = float(np.clip(init['Qg1'], Qg_min + 1e-4, Qg_max - 1e-4))

        # Building the model
        model = pyo.ConcreteModel()
        model.V1 = pyo.Var(bounds=(Vmin, Vmax), initialize=V1_init)
        model.V2 = pyo.Var(bounds=(Vmin, Vmax), initialize=V2_init)
        model.theta2 = pyo.Var(bounds=(-math.pi / 2, math.pi / 2), initialize=theta2_init)
        model.Qg1 = pyo.Var(bounds=(Qg_min, Qg_max), initialize=Qg1_init)

        # Constants
        Y = 1 / complex(R_pu, X_pu)
        G = float(Y.real)
        B = float(Y.imag)

        # Power flow equations
        def p_res1(m):
            return G * m.V1**2 - m.V1 * m.V2 * (G * pyo.cos(-m.theta2) + B * pyo.sin(-m.theta2)) - Pg1_pu == 0
        def q_res1(m):
            return -B * m.V1**2 - m.V1 * m.V2 * (G * pyo.sin(-m.theta2) - B * pyo.cos(-m.theta2)) - m.Qg1 == 0
        def p_res2(m):
            return G * m.V2**2 - m.V1 * m.V2 * (G * pyo.cos(-m.theta2) - B * pyo.sin(-m.theta2)) + P_load_pu == 0
        def q_res2(m):
            return -B * m.V2**2 + m.V1 * m.V2 * (G * pyo.sin(-m.theta2) + B * pyo.cos(-m.theta2)) + Q_load_pu == 0

        model.c1 = pyo.Constraint(rule=p_res1)
        model.c2 = pyo.Constraint(rule=q_res1)
        model.c3 = pyo.Constraint(rule=p_res2)
        model.c4 = pyo.Constraint(rule=q_res2)

        model.obj = pyo.Objective(expr=0.0, sense=pyo.minimize)

        # Using IPOPT for solving
        solver = SolverFactory('ipopt')
        solver.options['print_level'] = 0 if silent else 5
        solver.options['tol'] = ipopt_tol
        solver.options['acceptable_tol'] = max(ipopt_tol * 10, 1e-6)
        solver.options['max_iter'] = ipopt_max_iter
        
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
        print(f"Error in solve_with_ipopt: {e}")
        return False, None, None

# 单点求解
def true_two_stage_solver():
    """使用给定初始值点进行求解并绘制轨迹"""
    
    # 数据库设置
    db_path = 'tju_ipopt_results.db'
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE results
                    (base_V1 REAL, base_V2 REAL, feasible INTEGER, 
                     tju_loss REAL, ipopt_loss REAL, final_V1 REAL, final_V2 REAL,
                     tju_iters_used INTEGER, switched BOOLEAN)''')
    
    # 系统参数
    S_base = mpc["baseMVA"]
    Pg1_MW = 441.5
    Pg1_pu = Pg1_MW / S_base
    P_load_pu = mpc["bus"][1][2] / S_base
    Q_load_pu = mpc["bus"][1][3] / S_base
    R_pu = mpc["branch"][0][2]
    X_pu = mpc["branch"][0][3]
    Vmin, Vmax = 0.95, 1.05
    Qg_min, Qg_max = -4.0, 4.0
    
    # 初始值设置
    base_V1, base_V2 = 0.965, 1.02  # 您可以根据需要修改这两个值
    
    # 初始化变量
    V1, V2, theta2, Qg1 = initialize_variables(Pg1_pu)

    optimizer = TJU_v4(params=[V1, V2, theta2, Qg1], lr=0.01)
    
    # 记录轨迹
    trajectory = []
    
    # 阶段1: TJU优化
    for epoch in range(5000):
        # 前向计算
        P_res1, Q_res1, P_res2, Q_res2 = power_flow_eq(V1, V2, theta2, Pg1_pu, Qg1, P_load_pu, Q_load_pu, R_pu, X_pu)

        power_loss = (P_res1**2 + P_res2**2 + Q_res1**2 + Q_res2**2) * 100  # 动态归一
        voltage_loss = smooth_barrier(V1, Vmin, Vmax) + smooth_barrier(V2, Vmin, Vmax)
        qg_loss = smooth_barrier(Qg1, Qg_min, Qg_max)
        total_loss = power_loss + voltage_loss + qg_loss
        
        # 记录轨迹
        trajectory.append((V1.item(), V2.item()))

        # 反向传播与优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 检查是否达到收敛条件
        if total_loss.item() < 0.0001:
            print(f"收敛于第 {epoch + 1} 次迭代")
            break
        
    # 准备IPOPT初始值
    init_state = {
        'V1': V1.item(),
        'V2': V2.item(), 
        'theta2': theta2.item(),
        'Qg1': Qg1.item()
    }

    # 阶段2: IPOPT精修
    ipopt_ok = False
    final_V1, final_V2 = V1.item(), V2.item()
    
    ok, sol, _ = solve_with_ipopt(init=init_state, Pg1_pu=Pg1_pu, P_load_pu=P_load_pu,
                                   Q_load_pu=Q_load_pu, R_pu=R_pu, X_pu=X_pu,
                                   Vmin=Vmin, Vmax=Vmax, Qg_min=Qg_min, Qg_max=Qg_max)
                                   
    if ok and sol is not None:
        final_V1, final_V2 = sol['V1'], sol['V2']
        ipopt_ok = True
        print("IPOPT精修完成")
    else:
        print("IPOPT求解失败")

    # 可行性判断
    feasible = ipopt_ok

    # 存储结果
    cursor.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                   (base_V1, base_V2, int(feasible), 0, 0,  # tju_loss和ipopt_loss这里为0
                    final_V1, final_V2, epoch, int(ipopt_ok)))

    conn.commit()
    conn.close()
    
    # 绘制求解轨迹
    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', color='blue', label='Trajectory')
    plt.scatter(base_V1, base_V2, color='green', s=100, label='Initial Point', alpha=0.5)
    plt.scatter(final_V1, final_V2, color='red', s=100, label='Final Point', alpha=0.5)

    plt.title('TJU + IPOPT Trajectory for a Single Initial Point')
    plt.xlabel('V1 (p.u.)')
    plt.ylabel('V2 (p.u.)')
    plt.grid()
    plt.legend()
    plt.xlim(-4, 1.06)
    plt.ylim(0, 1.16)
    plt.show()

# 主入口
if __name__ == "__main__":
    true_two_stage_solver()