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
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# ===== 参数设置 - 添加负荷系数λ =====
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

# 负荷系数 - 用户可以修改这个值
load_factor = 1.0  # 默认值1.0，用户可以修改

# ===== 数据库初始化（表结构包含V1, V2, Qg1） =====
def init_db():
    # 删除旧数据库（如果存在）
    if os.path.exists('opf_results_tju_improved.db'):
        os.remove('opf_results_tju_improved.db')
        
    conn = sqlite3.connect('opf_results_tju_improved.db')
    cursor = conn.cursor()
    
    # 创建新表（包含Qg1字段）
    cursor.execute('''CREATE TABLE feasible_points
                    (Pg1_MW REAL, V1 REAL, V2 REAL, Qg1_Mvar REAL, loss REAL, epoch_count INTEGER)''')
    cursor.execute('''CREATE TABLE partial_results
                    (Pg1_MW REAL, best_loss REAL, epoch_count INTEGER)''')
    
    conn.commit()
    return conn

# ===== 智能初始化策略 =====
def initialize_variables(Pg1_pu):
    """基于Pg1线性插值生成物理合理的初值"""
    # 更平滑的经验公式
    base_V1 = 0.965 - 0.003 * (Pg1_pu - 4.4)
    base_V2 = 1.02 - 0.015 * (Pg1_pu - 4.4)
    return (
        torch.tensor([base_V1], requires_grad=True),
        torch.tensor([base_V2], requires_grad=True),
        torch.tensor([0.01], requires_grad=True),  # 更小的初始角度
        torch.tensor([0.0], requires_grad=True)    # 初始无功设为0
    )

# ===== 改进的约束处理 =====
def adaptive_constraint_loss(x, lower, upper, barrier_coef=0.05):
    """自适应约束处理函数"""
    # 计算约束违反程度
    lower_viol = torch.relu(lower - x)
    upper_viol = torch.relu(x - upper)
    
    # 线性惩罚 + 二次惩罚组合
    linear_penalty = torch.sum(lower_viol + upper_viol)
    quadratic_penalty = torch.sum(lower_viol**2 + upper_viol**2)
    
    return barrier_coef * (linear_penalty + quadratic_penalty)

# ===== 潮流方程计算（根据图示公式14实现） =====
def power_flow_eq(V1, V2, theta2, Pg1_pu, Qg1, P_load_pu, Q_load_pu, R_pu, X_pu, lambda_val):
    """
    根据图示公式(14)实现潮流方程：
    $$ \begin{cases}
    P_{\mathrm{G}, i}-\lambda P_{\mathrm{L}, i}-V_{i}\sum_{j\in i}V_{j}\left(G_{i j}\cos\theta_{i j}+B_{ij}\sin\theta_{i j}\right)=0\\
    Q_{\mathrm{G}, i}-\lambda Q_{\mathrm{L}, i}-V_{i}\sum_{j\in i}V_{j}\left(G_{i j}\sin\theta_{i j}-B_{ij}\cos\theta_{i j}\right)=0
    \end{cases} $$
    
    参数:
    lambda_val: 负荷系数λ（用户可设置）
    """
    # 导纳参数
    Y = 1 / complex(R_pu, X_pu)
    G_pu, B_line_pu = Y.real, Y.imag
    
    # 节点1 (发电机节点)
    # 节点1与节点2之间的相角差 (节点1为参考节点，相角为0)
    theta12 = -theta2
    
    # 节点1的注入功率计算（根据公式14右侧求和部分）
    P_inj1 = V1 * (V1 * (G_pu) + 
                   V2 * (G_pu * torch.cos(theta12) + B_line_pu * torch.sin(theta12)))
    
    Q_inj1 = V1 * (V1 * (-B_line_pu) + 
                   V2 * (G_pu * torch.sin(theta12) - B_line_pu * torch.cos(theta12)))
    
    # 节点2 (负荷节点)
    # 节点2与节点1之间的相角差（与节点1相反）
    theta21 = theta2
    
    # 节点2的注入功率计算（根据公式14右侧求和部分）
    P_inj2 = V2 * (V2 * (G_pu) + 
                   V1 * (G_pu * torch.cos(theta21) + B_line_pu * torch.sin(theta21)))
    
    Q_inj2 = V2 * (V2 * (-B_line_pu) + 
                   V1 * (G_pu * torch.sin(theta21) - B_line_pu * torch.cos(theta21)))
    
    # 根据图示公式(14)计算功率残差
    P_res1 = Pg1_pu - lambda_val * 0.0 - P_inj1  # 节点1没有负荷
    Q_res1 = Qg1 - lambda_val * 0.0 - Q_inj1      # 节点1没有负荷
    
    # 节点2没有发电机，但有负荷（应用负荷系数λ）
    P_res2 = 0.0 - lambda_val * P_load_pu - P_inj2
    Q_res2 = 0.0 - lambda_val * Q_load_pu - Q_inj2
    
    return P_res1, Q_res1, P_res2, Q_res2

# ===== 改进的TJU优化器 =====
class TJU_Improved(Optimizer):
    """针对OPF问题优化的TJU版本"""
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

                # 学习率调度（简化版）
                current_lr = group['lr']  # 固定学习率

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                # 更新动量项
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏置校正
                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                step_size = current_lr / bias_corr1

                # 近似Hessian处理
                delta_grad = grad - (exp_avg / bias_corr1)
                approx_hessian.mul_(group['beta_h']).addcmul_(
                    delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)
                else:
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(),
                                              torch.tensor(bound_val, device=p.device))

                # 组合二阶动量
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian,
                    alpha=1.0
                ).add_(group['eps'])

                # 计算更新方向
                update = exp_avg / denom

                # 执行参数更新
                p.add_(update, alpha=-step_size)

        return loss

# ===== 主计算流程 (改进版) =====
def compute_feasible_region_improved(lambda_val=1.0):
    """添加lambda_val参数，默认值为1.0"""
    print(f"计算开始，使用的负荷系数λ={lambda_val}")
    conn = init_db()
    S_base = mpc["baseMVA"]
    
    # 转换为标幺值
    P_load_pu = mpc["bus"][1][2] / S_base
    Q_load_pu = mpc["bus"][1][3] / S_base
    R_pu = mpc["branch"][0][2]
    X_pu = mpc["branch"][0][3]
    
    # 约束边界
    Vmin, Vmax = 0.95, 1.05
    Qg_min, Qg_max = -4.0, 4.0  # 标幺值
    
    # 扫描 Pg1（440-460 MW）
    num_points = 100
    Pg1_points_MW = np.linspace(540, 560, num_points)
    
    # 添加日志文件记录每个点的优化过程
    log_filename = f"optimization_log_lambda{lambda_val}.csv"
    with open(log_filename, "w") as log_file:
        log_file.write("Pg1_MW,Start_V1,Start_V2,Best_V1,Best_V2,Best_Qg1_Mvar,Final_Loss,Epochs,Lambda\n")
        
        for Pg1_MW in tqdm(Pg1_points_MW, desc=f"计算λ={lambda_val}"):
            Pg1_pu = Pg1_MW / S_base
            
            # 使用智能初始化
            V1, V2, theta2, Qg1 = initialize_variables(Pg1_pu)
            start_V1, start_V2 = V1.item(), V2.item()
            
            # 创建改进优化器
            optimizer = TJU_Improved(
                params=[V1, V2, theta2, Qg1],
                lr=0.01,
                betas=(0.9, 0.999),
                beta_h=0.85,
                weight_decay=0,
                use_cosine_scheduler=False
            )
            
            best_loss = float('inf')
            best_state = None
            start_time = time.time()
            
            # 迭代求解
            max_epochs = 20000  # 最大迭代次数
            converged = False
            min_change_counter = 0  # 用于跟踪损失变化率
            
            for epoch in range(max_epochs):
                # 前向计算潮流残差（应用负荷系数λ）
                P_res1, Q_res1, P_res2, Q_res2 = power_flow_eq(
                    V1, V2, theta2, Pg1_pu, Qg1, P_load_pu, Q_load_pu, R_pu, X_pu, lambda_val
                )
                
                # ===== 改进的损失函数 =====
                # 功率平衡约束（降低权重）
                power_loss = (P_res1**2 + Q_res1**2 + P_res2**2 + Q_res2**2) * 100
                
                # 电压约束（自适应约束处理）
                voltage_loss = adaptive_constraint_loss(V1, Vmin, Vmax) + adaptive_constraint_loss(V2, Vmin, Vmax)
                
                # 无功约束（自适应约束处理）
                qg_loss = adaptive_constraint_loss(Qg1, Qg_min, Qg_max)
                
                # 相角约束
                angle_loss = adaptive_constraint_loss(theta2, -math.pi, math.pi)
                
                # 总损失
                total_loss = power_loss + voltage_loss + qg_loss + angle_loss
                
                # 每1000步输出当前状态
                if epoch % 1000 == 0:
                    print(f"Pg1={Pg1_MW:.1f}MW, Epoch={epoch}, Loss={total_loss.item():.6f}, Qg1={Qg1.item()*S_base:.2f}Mvar")
                
                # 优化步骤
                optimizer.zero_grad()
                total_loss.backward()
                
                # ===== 梯度稳定性增强 =====
                # 对theta2梯度裁剪（防止三角函数梯度爆炸）
                if theta2.grad is not None:
                    theta2.grad = torch.clamp(theta2.grad, -100, 100)
                    
                optimizer.step()
                
                # 保留最佳解
                prev_best = best_loss
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_state = (V1.item(), V2.item(), theta2.item(), Qg1.item())
                    min_change_counter = 0
                else:
                    min_change_counter += 1
                
                # ===== 动态学习率衰减 =====
                if epoch > 0 and epoch % 2000 == 0:
                    for param_group in optimizer.param_groups:
                        new_lr = param_group['lr'] * 0.8
                        param_group['lr'] = max(new_lr, 1e-5)  # 设置最小学习率
                        print(f"学习率衰减至: {param_group['lr']}")
                
                # ===== 收敛检查 =====
                # 检查损失变化率
                if best_loss < 1e-3:
                    converged = True
                    break
                    
                # 检查损失平台期
                if min_change_counter > 500 and abs(prev_best - best_loss) < 1e-6:
                    converged = True
                    break
            
            end_time = time.time()
            compute_time = end_time - start_time
            
            # 获取最佳状态
            v1_val, v2_val, _, qg1_pu = best_state
            qg1_mvar = qg1_pu * S_base  # 转换为实际无功功率值
            
            # 记录优化过程（包含负荷系数λ）
            log_line = f"{Pg1_MW},{start_V1:.4f},{start_V2:.4f},{v1_val:.4f},{v2_val:.4f},{qg1_mvar:.4f},{best_loss:.6f},{epoch+1},{lambda_val}"
            log_file.write(log_line + "\n")
            
            # 约束检查（电压和无功功率）
            v1_in_range = Vmin <= v1_val <= Vmax
            v2_in_range = Vmin <= v2_val <= Vmax
            qg_in_range = Qg_min <= qg1_pu <= Qg_max
            
            # 保存可行解或部分结果
            if best_loss < 0.01 and v1_in_range and v2_in_range and qg_in_range:
                conn.execute(
                    "INSERT INTO feasible_points (Pg1_MW, V1, V2, Qg1_Mvar, loss, epoch_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (Pg1_MW, v1_val, v2_val, qg1_mvar, best_loss, epoch+1)
                )
                print(f"Pg1={Pg1_MW:.1f}MW找到可行解: V1={v1_val:.4f}, V2={v2_val:.4f}, Qg1={qg1_mvar:.2f}Mvar, "
                      f"损失={best_loss:.6f}, 迭代={epoch+1}, 用时={compute_time:.2f}秒")
            else:
                # 保存部分结果
                conn.execute(
                    "INSERT INTO partial_results (Pg1_MW, best_loss, epoch_count) VALUES (?, ?, ?)",
                    (Pg1_MW, best_loss, epoch+1)
                )
                reason = []
                if best_loss >= 0.01: reason.append("损失过高")
                if not v1_in_range: reason.append("V1越界")
                if not v2_in_range: reason.append("V2越界")
                if not qg_in_range: reason.append("Qg1越界")
                print(f"Pg1={Pg1_MW:.1f}MW未找到可行解! 原因: {'、'.join(reason)}, 最低损失={best_loss:.6f}, 迭代={epoch+1}, 用时={compute_time:.2f}秒")
                
            conn.commit()
    
    conn.close()
    print(f"计算完成，结果保存在 {log_filename} 和数据库 opf_results_tju_improved.db 中")

# ===== 可视化函数（去掉电压图，仅保留两个子图） =====
def visualize_results(lambda_val):
    # 确保数据库存在
    if not os.path.exists('opf_results_tju_improved.db'):
        print("数据库未找到，请先运行计算过程")
        return
        
    conn = sqlite3.connect('opf_results_tju_improved.db')
    c = conn.cursor()
    
    try:
        # 获取全部可行解
        c.execute("SELECT * FROM feasible_points")
        feasible_data = c.fetchall()
        
        # 获取部分结果
        c.execute("SELECT * FROM partial_results")
        partial_data = c.fetchall()
        
        conn.close()
        
        if not feasible_data and not partial_data:
            print("未找到任何结果！请查看日志文件分析原因")
            return
        
        # 创建图表布局（1行2列）
        plt.figure(figsize=(14, 6))
        
        # 图1：收敛损失分析
        plt.subplot(121)
        all_Pg1 = []
        all_loss = []
        
        # 处理部分结果数据
        if partial_data:
            partial_Pg1 = [row[0] for row in partial_data]
            partial_loss = [row[1] for row in partial_data]
            all_Pg1.extend(partial_Pg1)
            all_loss.extend(partial_loss)
            # 点大小设置为20
            plt.scatter(partial_Pg1, partial_loss, c='red', s=20, alpha=0.6, label="未收敛点")
        
        # 处理可行解数据
        if feasible_data:
            feasible_Pg1 = [row[0] for row in feasible_data]
            feasible_loss = [row[4] for row in feasible_data]
            all_Pg1.extend(feasible_Pg1)
            all_loss.extend(feasible_loss)
            # 点大小设置为20
            plt.scatter(feasible_Pg1, feasible_loss, c='green', s=20, alpha=0.6, label="可行解")
        
        if all_Pg1:
            plt.xlabel("发电机有功功率 Pg1 (MW)")
            plt.ylabel("最终损失值")
            plt.yscale('log')  # 对数坐标
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.title("不同Pg1下的收敛损失分析")
            plt.legend()
        else:
            plt.title("无优化结果")
        
        # 图2：Pg1-Qg1散点图（点大小设置为30）
        plt.subplot(122)
        if feasible_data:
            Pg1 = [row[0] for row in feasible_data]
            Qg1 = [row[3] for row in feasible_data]  # Qg1_Mvar
            
            
            plt.scatter(Pg1, Qg1, c='blue', s=10, alpha=0.7)
            plt.xlabel("发电机有功功率 Pg1 (MW)")
            plt.ylabel("发电机无功功率 Qg1 (Mvar)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.title(f"Pg1-Qg1 功率分布 (λ={lambda_val})")
            plt.xlim(400, 600)
            plt.ylim(0, 350)
            # 添加参考线
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        else:
            plt.title(f"无可行解 - Pg1-Qg1 (λ={lambda_val})")
        
        plt.tight_layout()
        plt.savefig(f"tju_results_lambda{lambda_val}.png", dpi=300)
        plt.show()
    
    except sqlite3.OperationalError as e:
        print(f"数据库查询错误: {e}")
        conn.close()
    except Exception as e:
        print(f"可视化过程中发生错误: {e}")
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    # 用户可在此处修改负荷系数λ值
    lambda_val = 1.0  # 设置您想要的负荷系数值，例如0.9, 1.0, 1.1
    
    # 计算可行域
    compute_feasible_region_improved(lambda_val)
    
    # 可视化结果
    visualize_results(lambda_val)