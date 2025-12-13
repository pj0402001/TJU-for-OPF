
import math
import numpy as np
import torch
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# 可选：提高数值稳定性
torch.set_default_dtype(torch.float64)

# ===============================
# 数据（沿用你提供的9节点系统）
# ===============================
baseMVA = 100.0

bus = np.array([
    [1, 3, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    [2, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    [3, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    [4, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    [5, 1, 54, 18, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    [6, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    [7, 1, 60, 21, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    [8, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
    [9, 1, 75, 30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
])

gen = np.array([
    [1, 0,   0, 300, -5, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 163, 0, 300, -5, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 85,  0, 300, -5, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

branch = np.array([
    [1, 4, 0,      0.0576, 0,     250, 250, 250, 0, 0, 1, -360, 360],
    [4, 5, 0.017,  0.092,  0.158, 150, 150, 150, 0, 0, 1, -360, 360],
    [5, 6, 0.039,  0.17,   0.358, 150, 150, 150, 0, 0, 1, -360, 360],
    [3, 6, 0,      0.0586, 0,     300, 300, 300, 0, 0, 1, -360, 360],
    [6, 7, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
    [7, 8, 0.0085, 0.072,  0.149, 250, 250, 250, 0, 0, 1, -360, 360],
    [8, 2, 0,      0.0625, 0,     250, 250, 250, 0, 0, 1, -360, 360],
    [8, 9, 0.032,  0.161,  0.306, 250, 250, 250, 0, 0, 1, -360, 360],
    [9, 4, 0.01,   0.085,  0.176, 250, 250, 250, 0, 0, 1, -360, 360],
])

# ============== 预处理 ==============
n_bus = bus.shape[0]
n_gen = gen.shape[0]

# 负荷、边界（p.u.）
Pd = np.array([float(bus[i, 2] / baseMVA) for i in range(n_bus)])
Qd = np.array([float(bus[i, 3] / baseMVA) for i in range(n_bus)])
Vmax = np.array([float(bus[i, 11]) for i in range(n_bus)])
Vmin = np.array([float(bus[i, 12]) for i in range(n_bus)])

# 发电机极限（p.u.）
gen_buses = [int(gen[i, 0]) - 1 for i in range(n_gen)]
gen_pmax = np.array([float(gen[i, 8] / baseMVA) for i in range(n_gen)])
gen_pmin = np.array([float(gen[i, 9] / baseMVA) for i in range(n_gen)])
gen_qmax = np.array([float(gen[i, 3] / baseMVA) for i in range(n_gen)])
gen_qmin = np.array([float(gen[i, 4] / baseMVA) for i in range(n_gen)])

# 映射：每个母线是否有发电机、其对应gen索引（如无为-1）
bus_has_gen = [False] * n_bus
gen_map = [-1] * n_bus   # bus -> gen index（若有）
for gi, b in enumerate(gen_buses):
    bus_has_gen[b] = True
    gen_map[b] = gi

# 构建 Ybus
def build_ybus():
    Ybus = np.zeros((n_bus, n_bus), dtype=complex)
    for br in branch:
        f = int(br[0]) - 1
        t = int(br[1]) - 1
        r, x, b = br[2:5]
        z = complex(r, x)
        if abs(z) > 1e-12:
            y = 1/z
        else:
            # 线路r=0的情况
            y = 1/complex(0, x)
        b_shunt = complex(0, b/2)

        Ybus[f, f] += y + b_shunt
        Ybus[t, t] += y + b_shunt
        Ybus[f, t] -= y
        Ybus[t, f] -= y
    return Ybus

Ybus = build_ybus()
G_np = Ybus.real
B_np = Ybus.imag

# ===============================
# 第一阶段：TJU 预解（Torch）
# ===============================
# 优化器（与你提供的TJU_Improved一致思路）
class TJU_Improved(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), beta_h=0.85, eps=1e-8,
                 rebound='constant', hessian_scale=0.05):
        defaults = dict(lr=lr, betas=betas, beta_h=beta_h, eps=eps,
                        rebound=rebound, hessian_scale=hessian_scale)
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
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)
                state['step'] += 1
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1**step
                bias_c2 = 1 - beta2**step
                step_size = group['lr'] / bias_c1

                delta_grad = grad - (exp_avg / bias_c1)
                approx_hessian.mul_(group['beta_h']).addcmul_(delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_h = approx_hessian.abs().clamp(min=1e-3)
                else:
                    denom_h = approx_hessian.abs().clamp(min=delta_grad.abs().max().item())

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)) + group['hessian_scale']*denom_h + group['eps']
                update = (exp_avg / bias_c1) / denom
                p.add_(update, alpha=-step_size)
        return loss

def softplus_barrier(x, lower, upper, coef=0.05, delta=0.02):
    # 对越界进行平滑惩罚，避免不可导
    sp = torch.nn.functional.softplus
    lower_viol = sp((lower - x)/delta)*delta
    upper_viol = sp((x - upper)/delta)*delta
    return coef * torch.sum(lower_viol + upper_viol)

def grad_norm(params):
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float((p.grad.detach()**2).sum().item())
    return math.sqrt(s)

def tju_stage_warm_start(p2_fixed_MW=None, p3_fixed_MW=None,
                         max_iters=5000, lr=0.01, verbose=False):
    """
    用TJ​​U生成暖启动初值。
    - p2_fixed_MW/p3_fixed_MW: 若给定，则用强罚项将Pg2/Pg3拉到该值
    返回 init dict（p.u.）: {'Vm','Va','Pg','Qg'}
    """
    device = torch.device('cpu')

    # 转 tensor（常量）
    G = torch.tensor(G_np, dtype=torch.get_default_dtype(), device=device)
    B = torch.tensor(B_np, dtype=torch.get_default_dtype(), device=device)
    Pd_t = torch.tensor(Pd, dtype=torch.get_default_dtype(), device=device)
    Qd_t = torch.tensor(Qd, dtype=torch.get_default_dtype(), device=device)
    Vmin_t = torch.tensor(Vmin, dtype=torch.get_default_dtype(), device=device)
    Vmax_t = torch.tensor(Vmax, dtype=torch.get_default_dtype(), device=device)
    pmin_t = torch.tensor(gen_pmin, dtype=torch.get_default_dtype(), device=device)
    pmax_t = torch.tensor(gen_pmax, dtype=torch.get_default_dtype(), device=device)
    qmin_t = torch.tensor(gen_qmin, dtype=torch.get_default_dtype(), device=device)
    qmax_t = torch.tensor(gen_qmax, dtype=torch.get_default_dtype(), device=device)

    # 初值：Vm=1.0，Va=0，Pg按pmax比例分配，总和≈负荷；Qg=0
    total_load_p = float(np.sum(Pd))
    if total_load_p <= 1e-8:
        total_load_p = 1.0
    cap = np.maximum(gen_pmax, 1e-6)
    share = cap / np.sum(cap)
    Pg0 = share * total_load_p
    # 若给定固定值（MW），则替换对应初值
    if p2_fixed_MW is not None:
        Pg0[1] = p2_fixed_MW / baseMVA
    if p3_fixed_MW is not None:
        Pg0[2] = p3_fixed_MW / baseMVA
    # 使Pg0在边界内
    Pg0 = np.minimum(np.maximum(Pg0, gen_pmin + 1e-6), gen_pmax - 1e-6)

    Vm = torch.ones(n_bus, requires_grad=True, device=device)
    Va_free = torch.zeros(n_bus-1, requires_grad=True, device=device)  # 参考母线角度固定0（bus 1）
    Pg = torch.tensor(Pg0, requires_grad=True, device=device)
    Qg = torch.zeros(n_gen, requires_grad=True, device=device)

    params = [Vm, Va_free, Pg, Qg]
    opt = TJU_Improved(params, lr=lr)

    # 目标固定罚权重
    fixed_w = 1000.0
    # 缩放
    scale_p = 1.0 / max(np.sum(np.abs(Pd)) + np.sum(np.abs(Pg0)) + 1e-2, 1.0)
    scale_q = 1.0 / max(np.sum(np.abs(Qd)) + 1.0, 1.0)

    best_loss = float('inf')
    best_state = None
    plateau = 0

    for it in range(max_iters):
        opt.zero_grad()

        Va = torch.cat([torch.zeros(1, device=device), Va_free])  # Va[0]=0
        # 角差矩阵
        theta = Va.unsqueeze(0) - Va.unsqueeze(1)  # [i,j] = Va_i - Va_j
        cos_th = torch.cos(theta)
        sin_th = torch.sin(theta)
        Vouter = Vm.unsqueeze(0) * Vm.unsqueeze(1)  # V_i V_j

        # 潮流计算
        Pcalc = torch.sum(Vouter * (G*cos_th + B*sin_th), dim=1)
        Qcalc = torch.sum(Vouter * (G*sin_th - B*cos_th), dim=1)

        # 发电注入映射到母线
        Pg_bus = torch.zeros(n_bus, dtype=Vm.dtype, device=device)
        Qg_bus = torch.zeros(n_bus, dtype=Vm.dtype, device=device)
        for b in range(n_bus):
            gi = gen_map[b]
            if gi >= 0:
                Pg_bus[b] = Pg[gi]
                Qg_bus[b] = Qg[gi]

        # 残差（Pg - Pd = Pcalc； Qg - Qd = Qcalc）
        Pres = (Pg_bus - Pd_t) - Pcalc
        Qres = (Qg_bus - Qd_t) - Qcalc

        power_loss = 200.0 * (scale_p*torch.sum(Pres**2) + scale_q*torch.sum(Qres**2))

        # 边界平滑罚
        barrier_coef = 0.05
        v_loss = softplus_barrier(Vm, Vmin_t, Vmax_t, coef=barrier_coef)
        p_loss = softplus_barrier(Pg, pmin_t, pmax_t, coef=barrier_coef)
        q_loss = softplus_barrier(Qg, qmin_t, qmax_t, coef=barrier_coef)
        # 相角边界（非参考母线）
        ang_lim = math.pi/3
        a_loss = softplus_barrier(Va_free, -ang_lim, ang_lim, coef=barrier_coef)

        # 可选固定P2/P3
        fix_pen = 0.0
        if p2_fixed_MW is not None:
            fix_pen = fix_pen + fixed_w*((Pg[1] - p2_fixed_MW/baseMVA)**2)
        if p3_fixed_MW is not None:
            fix_pen = fix_pen + fixed_w*((Pg[2] - p3_fixed_MW/baseMVA)**2)

        # 轻微正则（让Vm靠近1.0，Va靠近0）
        reg = 1e-3 * (torch.sum((Vm-1.0)**2) + torch.sum(Va_free**2) + 0.1*torch.sum(Pg**2) + 0.1*torch.sum(Qg**2))

        total_loss = power_loss + v_loss + p_loss + q_loss + a_loss + fix_pen + reg

        total_loss.backward()

        # 梯度裁剪（角度、Vm）
        if Va_free.grad is not None:
            Va_free.grad = torch.clamp(Va_free.grad, -100.0, 100.0)
        if Vm.grad is not None:
            Vm.grad = torch.clamp(Vm.grad, -100.0, 100.0)

        opt.step()

        # 记录最优
        if total_loss.item() < best_loss:
            best_loss = float(total_loss.item())
            best_state = {
                'Vm': Vm.detach().cpu().numpy().copy(),
                'Va': torch.cat([torch.zeros(1), Va_free.detach().cpu()]).numpy().copy(),
                'Pg': Pg.detach().cpu().numpy().copy(),
                'Qg': Qg.detach().cpu().numpy().copy(),
            }
            plateau = 0
        else:
            plateau += 1

        if verbose and it % 500 == 0:
            print(f"[TJU] iter={it}, loss={total_loss.item():.3e}, power={power_loss.item():.3e}")

        # 早停：梯度范数/平台
        gnorm = grad_norm(params)
        if gnorm < 1e-3 or plateau > 800 or best_loss < 1e-6:
            if verbose:
                print(f"[TJU] stop at iter={it}, best_loss={best_loss:.3e}, gnorm={gnorm:.3e}")
            break

        # 简单学习率衰减
        if it > 0 and it % 2000 == 0:
            for g in opt.param_groups:
                g['lr'] = max(g['lr']*0.8, 5e-5)

    # 返回最好状态
    if best_state is None:
        best_state = {
            'Vm': np.ones(n_bus),
            'Va': np.zeros(n_bus),
            'Pg': Pg0,
            'Qg': np.zeros(n_gen),
        }
    return best_state

# ===============================
# 第二阶段：IPOPT 精修（Pyomo）
# ===============================
def build_model_with_init(init_dict, p2_fixed_MW=None, p3_fixed_MW=None, reg_weight=1e-6):
    """
    使用第一阶段 init 作为初值构建 Pyomo 模型
    - reg_weight: 对偏离初值的小正则，有助于提高可解性与唯一性；设为0即“纯可行性”
    """
    Vm0 = init_dict['Vm']
    Va0 = init_dict['Va']
    Pg0 = init_dict['Pg']
    Qg0 = init_dict['Qg']

    model = pyo.ConcreteModel()
    model.buses = pyo.RangeSet(1, n_bus)
    model.gens = pyo.RangeSet(1, n_gen)

    # 变量与界
    def vm_bounds(m, i): return (Vmin[i-1], Vmax[i-1])
    model.Vm = pyo.Var(model.buses, bounds=vm_bounds, initialize=lambda m,i: float(Vm0[i-1]))

    # 角度界限可稍紧一些提升收敛
    ang_lim = math.pi/2
    model.Va = pyo.Var(model.buses, bounds=(-ang_lim, ang_lim), initialize=lambda m,i: float(Va0[i-1]))

    def pg_bounds(m, g): return (gen_pmin[g-1], gen_pmax[g-1])
    def qg_bounds(m, g): return (gen_qmin[g-1], gen_qmax[g-1])
    model.Pg = pyo.Var(model.gens, bounds=pg_bounds, initialize=lambda m,g: float(Pg0[g-1]))
    model.Qg = pyo.Var(model.gens, bounds=qg_bounds, initialize=lambda m,g: float(Qg0[g-1]))

    # 参考母线角度
    model.ref = pyo.Constraint(expr=model.Va[1] == 0.0)

    # AC 潮流约束
    G = G_np
    B = B_np
    def p_balance(m, i):
        i0 = i-1
        Pi = sum(m.Vm[i] * m.Vm[j] * (G[i0][j-1]*pyo.cos(m.Va[i]-m.Va[j]) + B[i0][j-1]*pyo.sin(m.Va[i]-m.Va[j]))
                 for j in m.buses)
        Pg_inj = m.Pg[gen_map[i0]+1] if bus_has_gen[i0] else 0.0
        return Pg_inj - Pd[i0] == Pi

    def q_balance(m, i):
        i0 = i-1
        Qi = sum(m.Vm[i] * m.Vm[j] * (G[i0][j-1]*pyo.sin(m.Va[i]-m.Va[j]) - B[i0][j-1]*pyo.cos(m.Va[i]-m.Va[j]))
                 for j in m.buses)
        Qg_inj = m.Qg[gen_map[i0]+1] if bus_has_gen[i0] else 0.0
        return Qg_inj - Qd[i0] == Qi

    model.p_bal = pyo.Constraint(model.buses, rule=p_balance)
    model.q_bal = pyo.Constraint(model.buses, rule=q_balance)

    # 可选固定P2/P3
    if p2_fixed_MW is not None:
        model.pg2_fix = pyo.Constraint(expr=model.Pg[2] == float(p2_fixed_MW/baseMVA))
    if p3_fixed_MW is not None:
        model.pg3_fix = pyo.Constraint(expr=model.Pg[3] == float(p3_fixed_MW/baseMVA))

    # 轻微正则化目标：接近初值（提高稳定性）。若你只做可行性，可把 reg_weight 设0
    if reg_weight > 0:
        obj_expr = 0.0
        for i in range(1, n_bus+1):
            obj_expr += (model.Vm[i] - float(Vm0[i-1]))**2 + 0.1*(model.Va[i] - float(Va0[i-1]))**2
        for g in range(1, n_gen+1):
            obj_expr += 0.1*(model.Pg[g] - float(Pg0[g-1]))**2 + 0.1*(model.Qg[g] - float(Qg0[g-1]))**2
        model.obj = pyo.Objective(expr=reg_weight*obj_expr, sense=pyo.minimize)
    else:
        model.obj = pyo.Objective(expr=0.0, sense=pyo.minimize)

    return model

def ipopt_solve(model, tee=False, max_iter=300, tol=1e-8):
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = int(max_iter)
    solver.options['tol'] = float(tol)
    solver.options['acceptable_tol'] = max(1e-6, float(tol)*10)
    solver.options['print_level'] = 0
    solver.options['mu_strategy'] = 'adaptive'
    solver.options['hessian_approximation'] = 'limited-memory'
    solver.options['warm_start_init_point'] = 'yes'
    solver.options['bound_relax_factor'] = 1e-8
    results = solver.solve(model, tee=tee)
    return results

# ===============================
# 一体化求解接口
# ===============================
def solve_9bus_two_stage(p2_fixed_MW=None, p3_fixed_MW=None, tju_iters=5000, tju_lr=0.01,
                         ipopt_max_iter=300, ipopt_tol=1e-8, verbose=False, reg_weight=1e-6):
    """
    双阶段：TJU 预解 -> IPOPT 精修
    输入：
      - p2_fixed_MW/p3_fixed_MW: 若设定，则固定发电机2/3有功输出（MW）
      - reg_weight: IPOPT目标的轻微正则权重，纯可行性可设0
    返回：
      - ok: 是否成功（optimal）
      - sol: 字典，包含 Vm, Va, Pg, Qg（均为物理量：Vm p.u., Va rad, P/Q 为 MW）
      - results: IPOPT返回对象
    """
    # 阶段1：TJU 预解
    init = tju_stage_warm_start(p2_fixed_MW=p2_fixed_MW, p3_fixed_MW=p3_fixed_MW,
                                max_iters=tju_iters, lr=tju_lr, verbose=verbose)

    # 阶段2：IPOPT 精修
    model = build_model_with_init(init, p2_fixed_MW=p2_fixed_MW, p3_fixed_MW=p3_fixed_MW, reg_weight=reg_weight)
    res = ipopt_solve(model, tee=verbose, max_iter=ipopt_max_iter, tol=ipopt_tol)

    ok = (res.solver.termination_condition == pyo.TerminationCondition.optimal)
    sol = None
    if ok:
        Vm = np.array([pyo.value(model.Vm[i]) for i in range(1, n_bus+1)])
        Va = np.array([pyo.value(model.Va[i]) for i in range(1, n_bus+1)])
        Pg = np.array([pyo.value(model.Pg[g])*baseMVA for g in range(1, n_gen+1)])  # MW
        Qg = np.array([pyo.value(model.Qg[g])*baseMVA for g in range(1, n_gen+1)])  # MVAr
        sol = {'Vm': Vm, 'Va': Va, 'Pg_MW': Pg, 'Qg_MVAr': Qg}
    return ok, sol, res

# ===============================
# 示例：直接运行
# ===============================
if __name__ == "__main__":
    # 示例1：不固定P2/P3（自由可行性）
    ok, sol, _ = solve_9bus_two_stage(p2_fixed_MW=None, p3_fixed_MW=None,
                                      tju_iters=4000, tju_lr=0.01,
                                      ipopt_max_iter=200, ipopt_tol=1e-8,
                                      verbose=True, reg_weight=1e-6)
    if ok:
        print("\n[结果] 自由可行性求解成功")
        print("Vm (p.u.):", np.round(sol['Vm'], 4))
        print("Va (rad): ", np.round(sol['Va'], 4))
        print("Pg (MW): ", np.round(sol['Pg_MW'], 2))
        print("Qg (MVAr):", np.round(sol['Qg_MVAr'], 2))
    else:
        print("\n[结果] 自由可行性求解失败")

    # 示例2：固定 P2=163 MW, P3=85 MW（与你原始数据一致）
    ok2, sol2, _ = solve_9bus_two_stage(p2_fixed_MW=163.0, p3_fixed_MW=85.0,
                                        tju_iters=4000, tju_lr=0.01,
                                        ipopt_max_iter=200, ipopt_tol=1e-8,
                                        verbose=True, reg_weight=1e-6)
    if ok2:
        print("\n[结果] 固定P2/P3求解成功")
        print("Vm (p.u.):", np.round(sol2['Vm'], 4))
        print("Va (rad): ", np.round(sol2['Va'], 4))
        print("Pg (MW): ", np.round(sol2['Pg_MW'], 2))
        print("Qg (MVAr):", np.round(sol2['Qg_MVAr'], 2))
    else:
        print("\n[结果] 固定P2/P3求解失败")
