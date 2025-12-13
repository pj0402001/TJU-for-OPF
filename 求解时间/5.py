import math
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from tqdm import tqdm
import csv
import time

# =========================
# 基本数据（WB5）
# =========================
baseMVA = 100.0

bus = np.array([
    [1, 3,   0,   0, 0, 0, 1, 1.0,   0.0, 345, 1, 1.13, 0.87],   # 你已改为 0.87-1.13
    [2, 1, 130,  20, 0, 0, 1, 1.0, -10.0, 345, 1, 1.13, 0.87],
    [3, 1, 130,  20, 0, 0, 1, 1.0, -20.0, 345, 1, 1.13, 0.87],
    [4, 1,  65,  10, 0, 0, 1, 1.0,-135.0, 345, 1, 1.13, 0.87],
    [5, 2,   0,   0, 0, 0, 1, 1.0,-140.0, 345, 1, 1.13, 0.87],
], dtype=float)

gen = np.array([
    [1, 500,  50, 1800,  -30, 1.0, 100, 1, 5000,   0],
    [5,   0,   0, 1800,  -30, 1.0, 100, 1, 5000,   0],
], dtype=float)

branch = np.array([
    [1, 2, 0.04, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [1, 3, 0.05, 0.10, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [2, 4, 0.55, 0.90, 0.45, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [3, 5, 0.55, 0.90, 0.45, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [4, 5, 0.06, 0.10, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [2, 3, 0.07, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
], dtype=float)

# =========================
# 预处理
# =========================
nb, nl, ng = bus.shape[0], branch.shape[0], gen.shape[0]
buses = list(range(1, nb+1))
lines = list(range(1, nl+1))
gens = list(range(1, ng+1))

gen_bus = {i+1: int(gen[i, 0]) for i in range(ng)}
Pmax = {i+1: gen[i, 8] / baseMVA for i in range(ng)}
Pmin = {i+1: gen[i, 9] / baseMVA for i in range(ng)}
Qmax = {i+1: gen[i, 3] / baseMVA for i in range(ng)}
Qmin = {i+1: gen[i, 4] / baseMVA for i in range(ng)}

Pd = {int(bus[i, 0]): bus[i, 2] / baseMVA for i in range(nb)}
Qd = {int(bus[i, 0]): bus[i, 3] / baseMVA for i in range(nb)}
Vmax = {int(bus[i, 0]): bus[i, 11] for i in range(nb)}
Vmin = {int(bus[i, 0]): bus[i, 12] for i in range(nb)}

fbus = {l + 1: int(branch[l, 0]) for l in range(nl)}
tbus = {l + 1: int(branch[l, 1]) for l in range(nl)}
r = {l + 1: branch[l, 2] for l in range(nl)}
x = {l + 1: branch[l, 3] for l in range(nl)}
b_total = {l + 1: branch[l, 4] for l in range(nl)}
rateA = {l + 1: branch[l, 5] / baseMVA for l in range(nl)}

def series_y(l):
    z = complex(r[l], x[l])
    y = 1.0 / z
    return y.real, y.imag

g_series = {l: series_y(l)[0] for l in lines}
b_series = {l: series_y(l)[1] for l in lines}

# Ybus
Ybus = np.zeros((nb, nb), dtype=complex)
for l in lines:
    i = fbus[l] - 1
    j = tbus[l] - 1
    y = 1.0 / complex(r[l], x[l])
    bsh = 1j * (b_total[l] / 2.0)
    Ybus[i, i] += y + bsh
    Ybus[j, j] += y + bsh
    Ybus[i, j] -= y
    Ybus[j, i] -= y
G = Ybus.real
B = Ybus.imag

swing_bus = 1

# =========================
# 可行性模型（目标为常数 0）
# =========================
def build_feas_model(pg5_MW, pg1_MW, init_type='flat', warm=None):
    m = pyo.ConcreteModel()
    m.BUS = pyo.Set(initialize=buses)
    m.GEN = pyo.Set(initialize=gens)
    m.LINE = pyo.Set(initialize=lines)

    m.Pd = pyo.Param(m.BUS, initialize=Pd)
    m.Qd = pyo.Param(m.BUS, initialize=Qd)
    m.Vmax = pyo.Param(m.BUS, initialize=Vmax)
    m.Vmin = pyo.Param(m.BUS, initialize=Vmin)

    m.Pmax = pyo.Param(m.GEN, initialize=Pmax, mutable=True)
    m.Pmin = pyo.Param(m.GEN, initialize=Pmin, mutable=True)
    m.Qmax = pyo.Param(m.GEN, initialize=Qmax, mutable=True)
    m.Qmin = pyo.Param(m.GEN, initialize=Qmin, mutable=True)

    m.fbus = pyo.Param(m.LINE, initialize=fbus, within=m.BUS)
    m.tbus = pyo.Param(m.LINE, initialize=tbus, within=m.BUS)
    m.g = pyo.Param(m.LINE, initialize=g_series)
    m.b = pyo.Param(m.LINE, initialize=b_series)
    m.bc = pyo.Param(m.LINE, initialize=b_total)
    m.rateA = pyo.Param(m.LINE, initialize=rateA)

    m.Vm = pyo.Var(m.BUS, within=pyo.Reals, initialize=1.0)
    m.Va = pyo.Var(m.BUS, within=pyo.Reals, initialize=0.0)
    m.Pg = pyo.Var(m.GEN, within=pyo.Reals, initialize=0.0)
    m.Qg = pyo.Var(m.GEN, within=pyo.Reals, initialize=0.0)

    def ref_angle_rule(_m, i):
        if i == swing_bus:
            return _m.Va[i] == 0.0
        return pyo.Constraint.Skip
    
    m.ref_angle = pyo.Constraint(m.BUS, rule=ref_angle_rule)

    for i in m.BUS:
        m.Vm[i].setlb(pyo.value(m.Vmin[i]))
        m.Vm[i].setub(pyo.value(m.Vmax[i]))

    for g in m.GEN:
        m.Pg[g].setlb(pyo.value(m.Pmin[g]))
        m.Pg[g].setub(pyo.value(m.Pmax[g]))
        m.Qg[g].setlb(pyo.value(m.Qmin[g]))
        m.Qg[g].setub(pyo.value(m.Qmax[g]))

    # 固定 Pg5、Pg1
    m.Pg[2].fix(pg5_MW / baseMVA)
    m.Pg[1].fix(pg1_MW / baseMVA)

    # 目标常数
    m.OBJ = pyo.Objective(expr=0.0, sense=pyo.minimize)

    # 潮流平衡
    Gmat, Bmat = G, B
    def P_balance_rule(_m, i):
        Pi = 0.0
        for j in _m.BUS:
            gij = Gmat[i - 1, j - 1]
            bij = Bmat[i - 1, j - 1]
            Pi += _m.Vm[i] * _m.Vm[j] * (gij * pyo.cos(_m.Va[i] - _m.Va[j]) + bij * pyo.sin(_m.Va[i] - _m.Va[j]))
        Pg_sum = sum(_m.Pg[g] for g in _m.GEN if gen_bus[g] == i)
        return Pi == Pg_sum - _m.Pd[i]
    
    m.P_balance = pyo.Constraint(m.BUS, rule=P_balance_rule)

    def Q_balance_rule(_m, i):
        Qi = 0.0
        for j in _m.BUS:
            gij = Gmat[i - 1, j - 1]
            bij = Bmat[i - 1, j - 1]
            Qi += _m.Vm[i] * _m.Vm[j] * (gij * pyo.sin(_m.Va[i] - _m.Va[j]) - bij * pyo.cos(_m.Va[i] - _m.Va[j]))
        Qg_sum = sum(_m.Qg[g] for g in _m.GEN if gen_bus[g] == i)
        return Qi == Qg_sum - _m.Qd[i]
    
    m.Q_balance = pyo.Constraint(m.BUS, rule=Q_balance_rule)

    # 线路热限（两端）
    m.Sf_limit = pyo.ConstraintList()
    m.St_limit = pyo.ConstraintList()
    for l in m.LINE:
        i = fbus[l]
        j = tbus[l]
        g = g_series[l]
        b = b_series[l]
        bc = b_total[l]
        def Pft(_m, i=i, j=j, g=g, b=b):
            return g * _m.Vm[i]**2 - g * _m.Vm[i] * _m.Vm[j] * pyo.cos(_m.Va[i] - _m.Va[j]) - b * _m.Vm[i] * _m.Vm[j] * pyo.sin(_m.Va[i] - _m.Va[j])

        def Qft(_m, i=i, j=j, g=g, b=b, bc=bc):
            return -(b + bc / 2.0) * _m.Vm[i]**2 + b * _m.Vm[i] * _m.Vm[j] * pyo.cos(_m.Va[i] - _m.Va[j]) - g * _m.Vm[i] * _m.Vm[j] * pyo.sin(_m.Va[i] - _m.Va[j])
        
        def Ptf(_m, i=i, j=j, g=g, b=b):
            return g * _m.Vm[j]**2 - g * _m.Vm[i] * _m.Vm[j] * pyo.cos(_m.Va[i] - _m.Va[j]) + b * _m.Vm[i] * _m.Vm[j] * pyo.sin(_m.Va[i] - _m.Va[j])
        
        def Qtf(_m, i=i, j=j, g=g, b=b, bc=bc):
            return -(b + bc / 2.0) * _m.Vm[j]**2 + b * _m.Vm[i] * _m.Vm[j] * pyo.cos(_m.Va[i] - _m.Va[j]) + g * _m.Vm[i] * _m.Vm[j] * pyo.sin(_m.Va[i] - _m.Va[j])

        m.Sf_limit.add(expr=Pft(m)**2 + Qft(m)**2 <= (rateA[l])**2)
        m.St_limit.add(expr=Ptf(m)**2 + Qtf(m)**2 <= (rateA[l])**2)

    # 初值设定
    for i in m.BUS:
        m.Vm[i].value = 1.0
        m.Va[i].value = 0.0

    m.Qg[1].value = 0.0
    m.Qg[2].value = 0.0

    if warm:
        for i in m.BUS:
            if 'Vm' in warm and i in warm['Vm']:
                m.Vm[i].value = warm['Vm'][i]
            if 'Va' in warm and i in warm['Va']:
                m.Va[i].value = warm['Va'][i]
        for g in m.GEN:
            if 'Qg' in warm and g in warm['Qg']:
                m.Qg[g].value = warm['Qg'][g]

    return m

def try_feasible(pg5_MW, pg1_MW, init_type='flat', warm=None):
    m = build_feas_model(pg5_MW, pg1_MW, init_type=init_type, warm=warm)
    
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-8
    solver.options['acceptable_tol'] = 1e-6
    solver.options['max_iter'] = 3000
    solver.options['print_level'] = 0

    start_time = time.time()  # 记录开始时间
    res = solver.solve(m, tee=False)
    elapsed_time = time.time() - start_time  # 计算求解时间

    term = res.solver.termination_condition
    stat = res.solver.status
    feasible = (stat in (pyo.SolverStatus.ok, pyo.SolverStatus.warning)) and \
               (term in (pyo.TerminationCondition.optimal,
                          pyo.TerminationCondition.locallyOptimal,
                          pyo.TerminationCondition.feasible))

    warm_next = None
    if feasible:
        warm_next = {
            'Vm': {i: pyo.value(m.Vm[i]) for i in m.BUS},
            'Va': {i: pyo.value(m.Va[i]) for i in m.BUS},
            'Qg': {g: pyo.value(m.Qg[g]) for g in m.GEN},
        }
    else:
        print(f"求解 PG5: {pg5_MW}, PG1: {pg1_MW} 时未找到可行解，时间: {elapsed_time:.3f}秒。")  # 打印未找到解的信息

    return feasible, warm_next, elapsed_time  # 返回计算时间

def check_feasible_multistart(pg5_MW, pg1_MW, warm_pools, direction):
    """
    warm_pools: dict key=(direction, init_type) -> warm_cache or None
    direction: 'inc' / 'dec'
    返回: feasible(bool), used_init_type(str or None), warm_updated
    """
    for init_type in ['flat', 'outer']:
        key = (direction, init_type)
        warm = warm_pools.get(key, None)
        ok, warm_new, elapsed_time = try_feasible(pg5_MW, pg1_MW, init_type=init_type, warm=warm)
        if ok:
            warm_pools[key] = warm_new  # 更新池子
            return True, init_type, warm_pools, elapsed_time
    warm_pools[(direction, 'flat')] = None
    warm_pools[(direction, 'outer')] = None
    return False, None, warm_pools, 0.0

# =========================
# 分阶段生成 PG1 点并扫描
# =========================
def gen_stage1_pg1():
    return np.arange(0.0, 700.0 + 1e-9, 5.0)

def gen_stage2_pg1(feasible_pg1_stage1):
    cand = set()
    for x in feasible_pg1_stage1:
        lo = max(0.0, x - 20.0)
        hi = min(700.0, x + 20.0)
        for y in np.arange(lo, hi + 1e-9, 1.0):
            cand.add(round(y, 3))
    return np.array(sorted(cand))

def gen_stage3_pg1():
    return np.arange(150.0, 250.0 + 1e-9, 0.5)

def scan_one_pg5(pg5_MW, enable_stage3=True):
    feasible_pg1 = set()
    warm_pools = {
        ('inc', 'flat'): None,
        ('inc', 'outer'): None,
        ('dec', 'flat'): None,
        ('dec', 'outer'): None,
    }

    pg1_list_inc = gen_stage1_pg1()
    pg1_list_dec = pg1_list_inc[::-1]

    for pg1 in pg1_list_inc:
        ok, _, warm_pools, elapsed_time = check_feasible_multistart(pg5_MW, pg1, warm_pools, 'inc')
        if ok:
            feasible_pg1.add(round(pg1, 3))

    for pg1 in pg1_list_dec:
        ok, _, warm_pools, elapsed_time = check_feasible_multistart(pg5_MW, pg1, warm_pools, 'dec')
        if ok:
            feasible_pg1.add(round(pg1, 3))

    if feasible_pg1:
        pg1_stage2 = gen_stage2_pg1(sorted(feasible_pg1))
        warm_pools = {k: None for k in warm_pools}
        for pg1 in pg1_stage2:
            ok, _, warm_pools, elapsed_time = check_feasible_multistart(pg5_MW, pg1, warm_pools, 'inc')
            if ok:
                feasible_pg1.add(round(pg1, 3))

        for pg1 in pg1_stage2[::-1]:
            ok, _, warm_pools, elapsed_time = check_feasible_multistart(pg5_MW, pg1, warm_pools, 'dec')
            if ok:
                feasible_pg1.add(round(pg1, 3))

    if enable_stage3:
        pg1_stage3 = gen_stage3_pg1()
        warm_pools = {k: None for k in warm_pools}
        for pg1 in pg1_stage3:
            ok, _, warm_pools, elapsed_time = check_feasible_multistart(pg5_MW, pg1, warm_pools, 'inc')
            if ok:
                feasible_pg1.add(round(pg1, 3))

        for pg1 in pg1_stage3[::-1]:
            ok, _, warm_pools, elapsed_time = check_feasible_multistart(pg5_MW, pg1, warm_pools, 'dec')
            if ok:
                feasible_pg1.add(round(pg1, 3))

    return sorted(feasible_pg1)

# =========================
# 主流程：单一进度条 + CSV + 绘图
# =========================
def main():
    pg5_vals = np.arange(0.0, 400.0 + 1e-9, 2.0)
    feasible_points = []

    for pg5 in tqdm(pg5_vals, desc="Scanning PG5 (multistart, 3-stage)", ncols=100):
        feas_pg1 = scan_one_pg5(pg5, enable_stage3=True)
        for pg1 in feas_pg1:
            feasible_points.append((pg5, pg1, 0.0))

    # 导出 CSV
    out_csv = "feasible_points_multistart.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PG5", "PG1", "loss"])
        for pg5, pg1, loss in feasible_points:
            w.writerow([f"{pg5:.6f}", f"{pg1:.6f}", f"{loss:.6f}"])
    print(f"CSV saved: {out_csv}, rows={len(feasible_points)}")

    # 绘图预览：横轴 PG1，纵轴 PG5
    if feasible_points:
        X = [p[1] for p in feasible_points]
        Y = [p[0] for p in feasible_points]
        plt.figure(figsize=(8, 6))
        plt.scatter(X, Y, s=6, c='tab:blue', alpha=0.75, edgecolors='none', label='Feasible points')
        plt.xlabel("PG1 (MW)")
        plt.ylabel("PG5 (MW)")
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No feasible points found. Consider relaxing steps or ranges.")

if __name__ == "__main__":
    main()