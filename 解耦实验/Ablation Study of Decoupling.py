import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from torch.optim import Optimizer

# ==========================================
# 1. 物理模型：LMBM3 (基于上传文件)
# ==========================================
class PowerFlowModelLMBM3:
    def __init__(self):
        self.baseMVA = 100.0
        # 支路参数 (From, To, R, X, B)
        self.branch_data = torch.tensor([
            [1, 3, 0.065, 0.620, 0.450],
            [3, 2, 0.025, 0.750, 0.700],
            [1, 2, 0.042, 0.900, 0.300],
        ], dtype=torch.float32)
        
        # 负载数据 (Bus 1, 2, 3) - 放大 1.5 倍重载
        self.bus_load = torch.tensor([
            [1.10, 0.40],
            [1.10, 0.40],
            [0.95, 0.50]
        ], dtype=torch.float32) * 1.5
        
        self.Ybus = self.build_ybus(3, self.branch_data)

    def build_ybus(self, nb, branch):
        Y = torch.zeros((nb, nb), dtype=torch.complex64)
        for line in branch:
            f, t = int(line[0])-1, int(line[1])-1
            y_s = 1 / complex(line[2], line[3])
            y_sh = complex(0, line[4]/2)
            Y[f, f] += y_s + y_sh; Y[t, t] += y_s + y_sh
            Y[f, t] -= y_s; Y[t, f] -= y_s
        return Y

    def compute_mismatch_and_cost(self, x_vars):
        """
        计算功率不平衡(Constraints) 和 发电成本(Cost)
        x_vars: [PG1, PG2, PG3, V1, V2, V3, th1, th2, th3]
        """
        PG = x_vars[0:3]
        V = x_vars[3:6]
        th = x_vars[6:9]
        
        # 1. 物理约束: 功率平衡 (Mismatch)
        Vc = V * torch.exp(1j * th)
        S_calc = Vc * torch.conj(torch.matmul(self.Ybus, Vc))
        
        # P_gen - P_load - P_inj = 0
        P_mis = PG - self.bus_load[:, 0] - S_calc.real
        Q_mis = -self.bus_load[:, 1] - S_calc.imag # 简化 Q 处理
        
        # 约束项范数 ||H(x)||^2
        constraint_norm = torch.sum(P_mis**2) + torch.sum(Q_mis**2)
        
        # 2. 目标函数: 发电成本 f(x) = sum(PG^2)
        # 这里用 PG^2 模拟二次成本函数
        cost_val = torch.sum(PG**2)
        
        return cost_val, constraint_norm

# ==========================================
# 2. 优化器 (Ablation: Coupled vs Decoupled)
# ==========================================
class TJU_Optimizer(Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), weight_decay=0.0, mode='coupled'):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, mode=mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            wd = group['weight_decay'] # 这里代表 Cost 权重
            mode = group['mode']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad # d(Penalty)/dx
                
                # --- Baseline: Coupled ---
                # 梯度 = d(Penalty)/dx + d(Cost)/dx
                # 直接将成本梯度加到物理梯度上，导致方向偏离
                if mode == 'coupled' and wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p); state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                
                # Adam 更新
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.sqrt().add_(1e-8)
                step_size = lr * math.sqrt(1 - beta2**state['step']) / (1 - beta1**state['step'])
                
                # 更新参数
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # --- NdaTJU: Decoupled ---
                # 物理梯度更新后，独立进行成本优化 (Decayed)
                # 这保证了每一步都主要沿物理可行方向走，同时收缩成本
                if mode == 'decoupled' and wd != 0:
                    p.mul_(1 - lr * wd)
        return loss

# ==========================================
# 3. 运行对比实验
# ==========================================
def run_experiment():
    # 读取背景数据
    try:
        df = pd.read_csv('lmbm3_feasible_points_v2_optimized.csv')
        feas_pg1 = df['PG1'].values
        feas_pg2 = df['PG2'].values
        print("LMBM3 Data Loaded.")
    except:
        print("Using Synthetic Background.")
        feas_pg1 = np.random.uniform(500, 520, 100)
        feas_pg2 = np.random.uniform(5, 10, 100)

    model = PowerFlowModelLMBM3()
    
    # 初始点: (250, 250)
    # 这是一个远离可行域的点，且需要大幅调整 P1, P2
    start_vals = torch.tensor([2.5, 2.5, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def run_opt(mode, wd, label):
        x = start_vals.clone().detach().requires_grad_(True)
        # 仅优化 PG (模拟 Cost), V/th 仅受约束驱动
        # 在 TJU 中通常全变量优化，这里对 PG 施加 WD (Cost)
        opt = TJU_Optimizer([x], lr=0.05, weight_decay=wd, mode=mode)
        
        traj_p1, traj_p2 = [], []
        energy_history = [] # 记录总能量 E
        rho = 50.0 # 罚因子 (Penalty Stiffness)
        
        for _ in range(300):
            opt.zero_grad()
            
            # 计算各项
            cost, constr_norm = model.compute_mismatch_and_cost(x)
            
            # 优化器使用的 "Loss" 仅为 Penalty 部分
            # Cost 部分通过 weight_decay 隐式或显式加入
            # 为了模拟论文：NdaTJU 中 Backward 仅传回 Penalty 梯度
            loss_backward = 0.5 * rho * constr_norm
            loss_backward.backward()
            opt.step()
            
            # 记录: 真实的论文定义能量 E = Cost + rho/2 * ||H||^2
            # 注意: 如果是 Coupled，Cost 被隐式加在梯度里，E 依然是总能量
            total_energy = cost.item() + 0.5 * rho * constr_norm.item()
            
            traj_p1.append(x[0].item() * 100)
            traj_p2.append(x[1].item() * 100)
            energy_history.append(total_energy)
            
        return traj_p1, traj_p2, energy_history

    # 强成本约束 (模拟追求经济性)
    WD = 0.1 
    
    print("Running Baseline...")
    p1_c, p2_c, e_c = run_opt('coupled', WD, 'Baseline')
    print("Running NdaTJU...")
    p1_d, p2_d, e_d = run_opt('decoupled', WD, 'NdaTJU')

    # --- 绘图：只生成能量函数 E(x) 的图 ---
    plt.figure(figsize=(8, 6))
    
    # 绘制能量函数收敛曲线
    plt.plot(e_c, 'k--', linewidth=2, label='Baseline E(x)')
    plt.plot(e_d, 'r-', linewidth=3, label='NdaTJU E(x)')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Total Energy Function $\mathcal{E}(x)$', fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('LMBM3_Energy_Function.png', dpi=300)
    print("Done.")
    plt.show()

if __name__ == '__main__':
    run_experiment()