import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Optimizer
# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# ==========================================
# 1. 优化器定义 (核心逻辑保持不变)
# ==========================================
class TJU_Comparison(Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            beta_h=0.85,
            eps=1e-8,
            rebound='constant',
            weight_decay=0.0,
            weight_decay_type='L2', # 'L2' (Baseline) 或 'AdamW' (NdaTJU)
            hessian_scale=0.05,
    ):
        defaults = dict(
            lr=lr, betas=betas, beta_h=beta_h, eps=eps, rebound=rebound,
            weight_decay=weight_decay, weight_decay_type=weight_decay_type,
            hessian_scale=hessian_scale
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
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                current_lr = group['lr']

                # --- Baseline: 耦合 L2 正则 ---
                # 直接修改梯度，干扰动量累积
                if group['weight_decay_type'] == 'L2' and group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                # 动量更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                step_size = current_lr / bias_corr1

                # 近似 Hessian
                delta_grad = grad - (exp_avg / bias_corr1)
                approx_hessian.mul_(group['beta_h']).addcmul_(
                    delta_grad, delta_grad, value=1 - group['beta_h'])
                
                denom_hessian = approx_hessian.abs().clamp_(min=1e-3)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian
                ).add_(group['eps'])

                update = exp_avg / denom

                # --- NdaTJU: 解耦权重衰减 ---
                # 独立于梯度更新，直接作用于参数
                if group['weight_decay_type'] == 'AdamW' and group['weight_decay'] != 0:
                    p.data.mul_(1 - group['weight_decay'] * current_lr)

                p.add_(update, alpha=-step_size)
        return loss

# ==========================================
# 2. 实验设置
# ==========================================
def run_experiment(wd_type, wd_value):
    # 初始化点：Rosenbrock 经典难点 (-1.5, 1.0)
    # 目标：(1, 1), Loss = 0
    x = torch.tensor([-1.5, 1.0], requires_grad=True)
    
    lr = 0.005 # 稍微降低学习率以凸显正则化影响
    steps = 800
    
    optimizer = TJU_Comparison(
        [x], 
        lr=lr, 
        weight_decay=wd_value,
        weight_decay_type=wd_type, 
        hessian_scale=0.1
    )
    
    min_loss = float('inf')
    
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        loss.backward()
        optimizer.step()
        
        # 记录过程中的最小 Loss，避免最后一步震荡影响判断
        if loss.item() < min_loss:
            min_loss = loss.item()
            
        # 简单早停
        if loss.item() < 1e-6:
            break
            
    return min_loss

# ==========================================
# 3. 运行精细扫描 (Fine-grained Scan)
# ==========================================
def main():
    # 生成对数分布的密集点：从 1e-6 到 1.0，共 50 个点
    wd_list = np.logspace(-6, 0, 50)
    
    losses_l2 = []     
    losses_ndatju = [] 
    
    print(f"正在扫描 {len(wd_list)} 个权重衰减参数点...")
    print(f"{'WD Value':<12} | {'Coupled':<10} | {'NdaTJU':<10}")
    print("-" * 40)
    
    for wd in wd_list:
        l2_val = run_experiment('L2', wd)
        ndatju_val = run_experiment('AdamW', wd)
        
        losses_l2.append(l2_val)
        losses_ndatju.append(ndatju_val)
        
        # 简单的进度打印
        # print(f"{wd:<12.1e} | {l2_val:<10.2e} | {ndatju_val:<10.2e}")

    # ==========================================
    # 4. 绘图 (增强版)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 绘制带填充的曲线
    plt.plot(wd_list, losses_l2, color='gray', linestyle='--', linewidth=2, label='Baseline')
    plt.plot(wd_list, losses_ndatju, color='#D32F2F', linewidth=3, label='NdaTJU ')
    
    # 填充差异区域，强调优势
    plt.fill_between(wd_list, losses_l2, losses_ndatju, 
                     where=(np.array(losses_l2) > np.array(losses_ndatju)),
                     color='red', alpha=0.1, interpolate=True, label='Robustness Advantage Region')

    plt.xscale('log')
    plt.yscale('log')
    
    # 坐标轴与标签
    plt.xlabel('Weight Decay Coefficient ($\lambda$)', fontsize=12, fontweight='bold')
    plt.ylabel('Minimum Loss Achieved (Log)', fontsize=12, fontweight='bold')
 
    
    # 辅助线
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(fontsize=11, loc='upper left')
    
    # 标注关键阈值
    # 找到 Baseline 开始失效的点 (Loss > 0.1)
    fail_idx = np.where(np.array(losses_l2) > 0.1)[0]
    if len(fail_idx) > 0:
        fail_wd = wd_list[fail_idx[0]]
        plt.axvline(x=fail_wd, color='gray', linestyle=':', alpha=0.8)
        plt.text(fail_wd, 1e-4, f'Baseline失效点\n$\lambda \\approx {fail_wd:.1e}$', rotation=90, va='bottom', color='gray')

    # 保存与显示
    plt.tight_layout()
    plt.savefig('FineGrained_Robustness_Comparison.png', dpi=300)
    print("\n高清对比图已保存为 FineGrained_Robustness_Comparison.png")
    plt.show()

if __name__ == '__main__':
    main()