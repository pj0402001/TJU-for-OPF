import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import math


class TJU_v4(Optimizer):
    r"""
    修正后的TJU_AdamW优化器，保持原TJU_v3精度的同时实现正确解耦权重衰减

    关键改进点：
    1. 修复权重衰减应用方式：使用当前学习率进行缩放 (current_lr * weight_decay)
    2. 保持原TJU_v3的近似Hessian处理逻辑
    3. 恢复原参数更新顺序，确保数值稳定性
    """

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
        # 参数校验（保持原有严格校验）
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if weight_decay_type not in ['L2', 'stable', 'AdamW']:  # 修正选项列表
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
                # 初始化状态（保持原TJU_v3结构）
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # ====== 学习率调度（保持原TJU_v3逻辑） ======
                current_lr = self._compute_lr(group, step)

                # ====== 核心参数更新 ======
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                # (1) L2正则化（保持原逻辑）
                if group['weight_decay_type'] == 'L2' and group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # (2) 更新动量项（保持原TJU_v3数值稳定性）
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # (3) 偏置校正（关键！保持原TJU_v3实现）
                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                step_size = current_lr / bias_corr1  # 合并学习率与一阶偏置校正

                # (4) 近似Hessian处理（保持原TJU_v3的clamp逻辑）
                delta_grad = grad - (exp_avg / bias_corr1)  # 修正后的梯度变化量
                approx_hessian.mul_(group['beta_h']).addcmul_(
                    delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)  # 保持原v3的clamp下限
                else:
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(),
                                              torch.tensor(bound_val, device=p.device))

                # (5) 组合二阶动量（保持原v3的混合逻辑）
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian,
                    alpha=1.0
                ).add_(group['eps'])

                # (6) 计算更新方向（关键修改点！恢复原v3的稳定性）
                update = exp_avg / denom

                # (7) 处理stable类型权重衰减（保持原v3逻辑）
                if group['weight_decay_type'] == 'stable' and group['weight_decay'] != 0:
                    decay_factor = group['weight_decay'] / denom.mean().clamp(min=1e-8)
                    update.add_(p, alpha=decay_factor)

                # ====== AdamW类型权重衰减（关键修正！）====== #
                # 在参数更新时应用解耦衰减（保持与当前学习率无关）
                if group['weight_decay_type'] == 'AdamW' and group['weight_decay'] != 0:
                    p.data.mul_(1 - group['weight_decay'] * current_lr)  # 与学习率解耦的关键修改！

                # (8) 执行参数更新（保持原v3的更新顺序）
                p.add_(update, alpha=-step_size)  # 注意：step_size已包含学习率和一阶偏置校正

        return loss

    def _compute_lr(self, group, step):
        """学习率调度（精确保持原TJU_v3实现）"""
        if step <= group['warmup']:
            return group['init_lr'] + (group['base_lr'] - group['init_lr']) * step / group['warmup']

        if not group['use_cosine_scheduler']:
            return group['base_lr']

        t = step - group['warmup']
        T = group['total_steps'] - group['warmup']
        if t <= T:
            return group['base_lr'] * (0.5 * (1 + math.cos(math.pi * t / T)))
        return group['base_lr'] * 0.01  # 保持原v3的后训练阶段学习率