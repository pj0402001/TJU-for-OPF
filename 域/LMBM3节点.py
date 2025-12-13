# filename: lmbm3_feasible_domain_v2_optimized.py
import math
import sqlite3
import csv
from typing import Dict, Tuple, Optional, List, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import pyomo.environ as pyo

# =========================
# LMBM3 (MATPOWER case) 数据
# =========================
baseMVA = 100.0

# bus: [bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va(deg), baseKV, zone, Vmax, Vmin]
bus = np.array([
    [1, 3, 110, 40, 0, 0, 1, 1.069,   0.000, 345, 1, 1.10, 0.90],
    [2, 2, 110, 40, 0, 0, 1, 1.028,   9.916, 345, 1, 1.10, 0.90],
    [3, 2,  95, 50, 0, 0, 1, 1.001, -13.561, 345, 1, 1.10, 0.90],
], dtype=float)

# gen: [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin]
gen = np.array([
    [1, 131.09, 17.02, 10000, -1000, 1.069, 100, 1, 10000, 0],
    [2, 185.93, -3.50,  1000, -1000, 1.028, 100, 1, 10000, 0],
    [3,   0.00,  0.06,  1000, -1000, 1.001, 100, 1,     0, 0],
], dtype=float)

# branch: [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]
branch = np.array([
    [1, 3, 0.065, 0.620, 0.450, 9999, 9999, 9999, 0, 0, 1, -360, 360],
    [3, 2, 0.025, 0.750, 0.700,  186, 9999, 9999, 0, 0, 1, -360, 360],  # 关键热限
    [1, 2, 0.042, 0.900, 0.300, 9999, 9999, 9999, 0, 0, 1, -360, 360],
], dtype=float)

# =========================
# 全局设置（优化后）
# =========================
LOAD_FACTOR = 1.50   # Pd/Qd 放大进入潮流

# 扫描上限（MW）
PG1_MAX = 600.0
PG2_MAX = 500.0

# Stage-1（粗扫） - 增大步长减少点数
STAGE1_STEP_PG2 = 1.0  # 从1.0增大到5.0
MID_PG2_REFINE = (200.0, 380.0)  # 中段加密以帮助发现第二分支
STAGE1_STEP_PG2_MID = 2  # 从5.0增大到10.0

STAGE1_STEP_PG1_WIDE = 2  # 从5.0增大到10.0
STAGE1_STEP_PG1_CENTER = 5.0  # 保持5.0
STAGE1_CENTER_HALF_WIDTH = 200.0  # 不变

# 区段生长（沿 PG1 向两侧扩展） - 减少扩展范围
ENABLE_REGION_GROW = True
RG_STEP = 2.0            # 从1.0增大到2.0
RG_MAX_SPAN = 100.0      # 从600.0减小到100.0

# 随机重启（多分支激活） - 减少尝试次数
RAND_TRIES_DEFAULT = 1   # 从2减小到1
RAND_TRIES_MID = 1       # 保持不变
VM_MARGIN = 0.02         # 随机 Vm 与上下限的安全边距

# =========================
# 预处理
# =========================
nb, nl, ng = bus.shape[0], branch.shape[0], gen.shape[0]
buses = list(range(1, nb + 1))
lines = list(range(1, nl + 1))
gens = list(range(1, ng + 1))

gen_bus = {i + 1: int(gen[i, 0]) for i in range(ng)}
Pmax = {i + 1: gen[i, 8] / baseMVA for i in range(ng)}
Pmin = {i + 1: gen[i, 9] / baseMVA for i in range(ng)}
Qmax = {i + 1: gen[i, 3] / baseMVA for i in range(ng)}
Qmin = {i + 1: gen[i, 4] / baseMVA for i in range(ng)}

Pd0 = {int(bus[i, 0]): bus[i, 2] / baseMVA for i in range(nb)}
Qd0 = {int(bus[i, 0]): bus[i, 3] / baseMVA for i in range(nb)}
Pd = {k: Pd0[k] * LOAD_FACTOR for k in Pd0}
Qd = {k: Qd0[k] * LOAD_FACTOR for k in Qd0}

Vmax = {int(bus[i, 0]): bus[i, 11] for i in range(nb)}
Vmin = {int(bus[i, 0]): bus[i, 12] for i in range(nb)}
Vm_init_from_mpc = {int(bus[i, 0]): bus[i, 7] for i in range(nb)}
Va_init_from_mpc = {int(bus[i, 0]): math.radians(bus[i, 8]) for i in range(nb)}

fbus = {l + 1: int(branch[l, 0]) for l in range(nl)}
tbus = {l + 1: int(branch[l, 1]) for l in range(nl)}
r = {l + 1: branch[l, 2] for l in range(nl)}
x = {l + 1: branch[l, 3] for l in range(nl)}
b_total = {l + 1: branch[l, 4] for l in range(nl)}
rateA = {l + 1: branch[l, 5] / baseMVA for l in range(nl)}

def series_y(l: int):
    z = complex(r[l], x[l])
    y = 1.0 / z
    return y.real, y.imag

g_series = {l: series_y(l)[0] for l in lines}
b_series = {l: series_y(l)[1] for l in lines}

# 构建 Ybus
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
# 建模（可行性，Objective=0）
# =========================
def build_feas_model(pg1_MW: float, pg2_MW: float,
                     init_type: str = 'flat',
                     warm: Optional[Dict] = None,
                     rand_key: Optional[int] = None) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.BUS = pyo.Set(initialize=buses)
    m.GEN = pyo.Set(initialize=gens)
    m.LINE = pyo.Set(initialize=lines)

    # 参数
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

    # 变量
    m.Vm = pyo.Var(m.BUS, within=pyo.Reals)
    m.Va = pyo.Var(m.BUS, within=pyo.Reals)
    m.Pg = pyo.Var(m.GEN, within=pyo.Reals)
    m.Qg = pyo.Var(m.GEN, within=pyo.Reals)

    # 参考角
    def ref_angle_rule(_m, i):
        if i == swing_bus:
            return _m.Va[i] == 0.0
        return pyo.Constraint.Skip
    m.ref_angle = pyo.Constraint(m.BUS, rule=ref_angle_rule)

    # 电压限
    for i in m.BUS:
        m.Vm[i].setlb(pyo.value(m.Vmin[i]))
        m.Vm[i].setub(pyo.value(m.Vmax[i]))

    # 发电机上下限
    for g in m.GEN:
        m.Pg[g].setlb(pyo.value(m.Pmin[g]))
        m.Pg[g].setub(pyo.value(m.Pmax[g]))
        m.Qg[g].setlb(pyo.value(m.Qmin[g]))
        m.Qg[g].setub(pyo.value(m.Qmax[g]))

    # 固定 Pg1、Pg2、Pg3
    m.Pg[1].fix(pg1_MW / baseMVA)
    m.Pg[2].fix(pg2_MW / baseMVA)
    m.Pg[3].fix(0.0)

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
        i = fbus[l]; j = tbus[l]
        g = g_series[l]; b = b_series[l]; bc = b_total[l]
        def Pft(_m, i=i, j=j, g=g, b=b):
            return g*_m.Vm[i]**2 - g*_m.Vm[i]*_m.Vm[j]*pyo.cos(_m.Va[i]-_m.Va[j]) - b*_m.Vm[i]*_m.Vm[j]*pyo.sin(_m.Va[i]-_m.Va[j])
        def Qft(_m, i=i, j=j, g=g, b=b, bc=bc):
            return -(b + bc/2.0)*_m.Vm[i]**2 + b*_m.Vm[i]*_m.Vm[j]*pyo.cos(_m.Va[i]-_m.Va[j]) - g*_m.Vm[i]*_m.Vm[j]*pyo.sin(_m.Va[i]-_m.Va[j])
        def Ptf(_m, i=i, j=j, g=g, b=b):
            return g*_m.Vm[j]**2 - g*_m.Vm[i]*_m.Vm[j]*pyo.cos(_m.Va[i]-_m.Va[j]) + b*_m.Vm[i]*_m.Vm[j]*pyo.sin(_m.Va[i]-_m.Va[j])
        def Qtf(_m, i=i, j=j, g=g, b=b, bc=bc):
            return -(b + bc/2.0)*_m.Vm[j]**2 + b*_m.Vm[i]*_m.Vm[j]*pyo.cos(_m.Va[i]-_m.Va[j]) + g*_m.Vm[i]*_m.Vm[j]*pyo.sin(_m.Va[i]-_m.Va[j])
        m.Sf_limit.add(expr=Pft(m)**2 + Qft(m)**2 <= (rateA[l])**2)
        m.St_limit.add(expr=Ptf(m)**2 + Qtf(m)**2 <= (rateA[l])**2)

    # 初值
    for i in m.BUS:
        m.Vm[i].value = 1.0
        m.Va[i].value = 0.0
    m.Qg[1].value = 0.0
    m.Qg[2].value = 0.0
    m.Qg[3].value = 0.0

    if init_type == 'outer':
        for i in m.BUS:
            m.Vm[i].value = Vm_init_from_mpc[i]
            m.Va[i].value = Va_init_from_mpc[i]
    elif init_type == 'lowV':
        for i in m.BUS:
            m.Vm[i].value = max(Vmin[i] + 0.02, 0.92)
            m.Va[i].value = 0.0
    elif init_type == 'hiAngle':
        m.Va[2].value = math.radians(+80.0)
        m.Va[3].value = math.radians(-80.0)
        m.Vm[2].value = 1.0; m.Vm[3].value = 1.0
    elif init_type == 'negHiAngle':
        m.Va[2].value = math.radians(-80.0)
        m.Va[3].value = math.radians(+80.0)
        m.Vm[2].value = 1.0; m.Vm[3].value = 1.0
    elif init_type == 'rand':
        rng = np.random.default_rng(rand_key if rand_key is not None else None)
        for i in m.BUS:
            lo = max(Vmin[i] + VM_MARGIN, 0.90)
            hi = min(Vmax[i] - VM_MARGIN, 1.10)
            m.Vm[i].value = float(rng.uniform(lo, hi))
            # Va1 固定为 0（参考），其余随机
            if i != swing_bus:
                m.Va[i].value = float(rng.uniform(-math.pi, math.pi))

    # 覆盖热启动
    if warm:
        for i in m.BUS:
            if 'Vm' in warm and i in warm['Vm']: m.Vm[i].value = warm['Vm'][i]
            if 'Va' in warm and i in warm['Va']: m.Va[i].value = warm['Va'][i]
        for g in m.GEN:
            if 'Qg' in warm and g in warm['Qg']: m.Qg[g].value = warm['Qg'][g]

    return m

def solve_ipopt(m: pyo.ConcreteModel) -> Tuple[bool, Optional[Dict]]:
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-8
    solver.options['acceptable_tol'] = 1e-6
    solver.options['max_iter'] = 3000
    solver.options['print_level'] = 0
    # 禁止输出不可行警告
    solver.options['output_file'] = 'ipopt_out.log'  # 重定向输出到文件
    solver.options['print_user_options'] = 'no'
    solver.options['print_options_documentation'] = 'no'
    
    res = solver.solve(m, tee=False)
    term = res.solver.termination_condition
    stat = res.solver.status
    feasible = (stat in (pyo.SolverStatus.ok, pyo.SolverStatus.warning)) and \
               (term in (pyo.TerminationCondition.optimal,
                         pyo.TerminationCondition.locallyOptimal,
                         pyo.TerminationCondition.feasible))
    if not feasible:
        return False, None
    sol = {
        'Vm': {i: pyo.value(m.Vm[i]) for i in m.BUS},
        'Va': {i: pyo.value(m.Va[i]) for i in m.BUS},
        'Pg': {g: pyo.value(m.Pg[g]) for g in m.GEN},
        'Qg': {g: pyo.value(m.Qg[g]) for g in m.GEN},
    }
    return True, sol

def try_feasible(pg1_MW: float, pg2_MW: float,
                 init_type: str = 'flat',
                 warm: Optional[Dict] = None,
                 rand_key: Optional[int] = None) -> Tuple[bool, Optional[Dict]]:
    m = build_feas_model(pg1_MW, pg2_MW, init_type=init_type, warm=warm, rand_key=rand_key)
    return solve_ipopt(m)

def check_feasible_multistart(pg2_MW: float, pg1_MW: float,
                              warm_pools: Dict[Tuple[str, str], Optional[Dict]],
                              direction: str,
                              rand_tries: int = 0) -> Tuple[bool, Optional[Dict], Dict]:
    """
    多起点+多方向，加入高角差与随机重启。
    """
    # 简化初始点策略
    init_order = ['flat', 'outer', 'hiAngle']
    for init_type in init_order:
        key = (direction, init_type)
        warm = warm_pools.get(key, None)
        ok, sol = try_feasible(pg1_MW, pg2_MW, init_type=init_type, warm=warm)
        if ok:
            warm_pools[key] = sol
            return True, sol, warm_pools

    # 随机重启（不使用热启动）
    for k in range(rand_tries):
        ok, sol = try_feasible(pg1_MW, pg2_MW, init_type='rand', warm=None, rand_key=k + 1000)
        if ok:
            warm_pools[(direction, 'flat')] = sol
            return True, sol, warm_pools

    # 全失败，清空该方向所有池
    for t in init_order:
        warm_pools[(direction, t)] = None
    return False, None, warm_pools

# =========================
# 分阶段 PG1 网格（对固定 PG2）
# =========================
def gen_stage1_pg1(center: float) -> np.ndarray:
    s1 = np.arange(0.0, PG1_MAX + 1e-9, STAGE1_STEP_PG1_WIDE)
    lo = max(0.0, center - STAGE1_CENTER_HALF_WIDTH)
    hi = min(PG1_MAX, center + STAGE1_CENTER_HALF_WIDTH)
    s2 = np.arange(lo, hi + 1e-9, STAGE1_STEP_PG1_CENTER)
    merged = sorted(set([round(x, 3) for x in np.concatenate([s1, s2])]))
    return np.array(merged, dtype=float)

# =========================
# 区段生长（沿 PG1 向两侧扩展）
# =========================
def region_grow(pg2_MW: float, seed_pg1: float, seed_sol: Dict,
                direction_label: str,  # 'inc' or 'dec'
                step: float = RG_STEP,
                max_span: float = RG_MAX_SPAN) -> List[Tuple[float, Dict]]:
    results = [(seed_pg1, seed_sol)]
    warm = seed_sol.copy()
    curr = seed_pg1
    span = 0.0
    
    # 只向一个方向生长（根据标签）
    dir_sign = 1 if direction_label == 'inc' else -1
    
    while span < max_span:
        nxt = curr + dir_sign * step
        if nxt < 0.0 or nxt > PG1_MAX:
            break
        ok, sol = try_feasible(nxt, pg2_MW, init_type='flat', warm=warm)
        if not ok:
            break
        results.append((nxt, sol))
        warm = sol
        curr = nxt
        span += step
        
    return results

# =========================
# 扫描：固定 PG2，寻找可行 PG1
# =========================
def scan_one_pg2(pg2_MW: float, total_load_MW: float) -> List[Dict]:
    feasible_records: List[Dict] = []

    warm_pools = {
        ('inc', 'flat'): None, ('inc', 'outer'): None, ('inc', 'hiAngle'): None,
        ('dec', 'flat'): None, ('dec', 'outer'): None, ('dec', 'hiAngle'): None,
    }

    center = total_load_MW - pg2_MW

    # 针对中段 PG2 增加随机重启次数
    in_mid = (MID_PG2_REFINE[0] - 1e-6 <= pg2_MW <= MID_PG2_REFINE[1] + 1e-6)
    rand_tries = RAND_TRIES_MID if in_mid else RAND_TRIES_DEFAULT

    # 阶段一：粗扫（两个方向）
    pg1_stage1 = gen_stage1_pg1(center)
    seeds: List[Tuple[float, Dict]] = []

    # 优化：只扫描一次，避免重复计算
    for pg1 in pg1_stage1:
        # 尝试增加方向
        ok, sol, warm_pools = check_feasible_multistart(pg2_MW, pg1, warm_pools, 'inc', rand_tries=rand_tries)
        if ok and sol is not None:
            seeds.append((pg1, sol))
            feasible_records.append(pack_record(pg1, pg2_MW, sol))
        else:
            # 尝试减少方向
            ok, sol, warm_pools = check_feasible_multistart(pg2_MW, pg1, warm_pools, 'dec', rand_tries=rand_tries)
            if ok and sol is not None:
                seeds.append((pg1, sol))
                feasible_records.append(pack_record(pg1, pg2_MW, sol))

    # 区段生长：从已发现的种子沿 PG1 扩展
    if ENABLE_REGION_GROW and seeds:
        grown_pg1 = set(round(s[0], 6) for s in seeds)
        for seed_pg1, seed_sol in seeds:
            # 根据种子位置决定生长方向
            if seed_pg1 < center:
                direction = 'inc'
            else:
                direction = 'dec'
                
            grown = region_grow(pg2_MW, seed_pg1, seed_sol, direction, step=RG_STEP, max_span=RG_MAX_SPAN)
            for pg1_val, sol in grown:
                k = round(pg1_val, 6)
                if k not in grown_pg1:
                    feasible_records.append(pack_record(pg1_val, pg2_MW, sol))
                    grown_pg1.add(k)

    # 去重（按 PG1,PG2）
    uniq = {}
    for r in feasible_records:
        key = (round(r['PG1'], 6), round(r['PG2'], 6))
        if key not in uniq:
            uniq[key] = r
    return list(uniq.values())

def pack_record(pg1_MW: float, pg2_MW: float, sol: Dict) -> Dict:
    pg1_pu = sol['Pg'][1]; pg2_pu = sol['Pg'][2]; pg3_pu = sol['Pg'][3]
    sumPg_MW = baseMVA * (pg1_pu + pg2_pu + pg3_pu)
    total_load_MW = baseMVA * sum(Pd[i] for i in buses)
    Ploss = sumPg_MW - total_load_MW
    return {
        'PG1': pg1_MW,
        'PG2': pg2_MW,
        'PG3': 0.0,
        'QG1': baseMVA * sol['Qg'][1],
        'QG2': baseMVA * sol['Qg'][2],
        'QG3': baseMVA * sol['Qg'][3],
        'Vm1': sol['Vm'][1], 'Vm2': sol['Vm'][2], 'Vm3': sol['Vm'][3],
        'Va1_deg': math.degrees(sol['Va'][1]),
        'Va2_deg': math.degrees(sol['Va'][2]),
        'Va3_deg': math.degrees(sol['Va'][3]),
        'Ploss_MW': Ploss,
        'load_factor': LOAD_FACTOR
    }

# =========================
# 主流程
# =========================
def main():
    total_load_MW = baseMVA * sum(Pd[i] for i in buses)

    # PG2 网格：全域 10 MW 步，并在[200,380] 以 5 MW 加密
    pg2_full = np.arange(0.0, PG2_MAX + 1e-9, STAGE1_STEP_PG2)
    pg2_mid = np.arange(MID_PG2_REFINE[0], MID_PG2_REFINE[1] + 1e-9, STAGE1_STEP_PG2_MID)
    pg2_vals = np.array(sorted(set([round(x, 3) for x in np.concatenate([pg2_full, pg2_mid])])))

    all_records: List[Dict] = []
    for pg2 in tqdm(pg2_vals, desc="Scanning PG2", ncols=100):
        recs = scan_one_pg2(pg2, total_load_MW)
        all_records.extend(recs)

    if not all_records:
        print("未找到可行点，请调整参数或步长。")
        return

    df = pd.DataFrame(all_records)
    df.sort_values(by=['PG2', 'PG1'], inplace=True, ignore_index=True)

    # 写入 SQLite
    db_path = "lmbm3_feasible_v2_optimized.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feasible_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            PG1 REAL, PG2 REAL, PG3 REAL,
            QG1 REAL, QG2 REAL, QG3 REAL,
            Vm1 REAL, Vm2 REAL, Vm3 REAL,
            Va1_deg REAL, Va2_deg REAL, Va3_deg REAL,
            Ploss_MW REAL,
            load_factor REAL
        )
    """)
    cur.execute("DELETE FROM feasible_points")
    conn.commit()
    cur.executemany("""
        INSERT INTO feasible_points
        (PG1, PG2, PG3, QG1, QG2, QG3, Vm1, Vm2, Vm3, Va1_deg, Va2_deg, Va3_deg, Ploss_MW, load_factor)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (r['PG1'], r['PG2'], r['PG3'],
         r['QG1'], r['QG2'], r['QG3'],
         r['Vm1'], r['Vm2'], r['Vm3'],
         r['Va1_deg'], r['Va2_deg'], r['Va3_deg'],
         r['Ploss_MW'], r['load_factor'])
        for r in all_records
    ])
    conn.commit()
    conn.close()
    print(f"SQLite 数据库已写入：{db_path}，共 {len(all_records)} 条记录")

    # 导出 CSV
    csv_path = "lmbm3_feasible_points_v2_optimized.csv"
    cols = ['PG1', 'PG2', 'PG3', 'QG1', 'QG2', 'QG3', 'Vm1', 'Vm2', 'Vm3', 'Va1_deg', 'Va2_deg', 'Va3_deg', 'Ploss_MW', 'load_factor']
    df.to_csv(csv_path, index=False, columns=cols, float_format="%.6f", encoding="utf-8")
    print(f"CSV 已导出：{csv_path}")

    # 图1：PG1–PG2 可行散点
    plt.figure(figsize=(7.5, 6))
    plt.scatter(df['PG1'], df['PG2'], s=8, c='tab:blue', alpha=0.75, edgecolors='none', label='Feasible')
    plt.xlabel("PG1 (MW)")
    plt.ylabel("PG2 (MW)")
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig("feasible_domain.png", dpi=300)
    plt.show()

    # 图2：QG1–QG2–QG3 三维散点
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['QG1'], df['QG2'], df['QG3'], s=8, c='tab:blue', alpha=0.85, depthshade=True)
    ax.set_xlabel("QG1 (MVAr)")
    ax.set_ylabel("QG2 (MVAr)")
    ax.set_zlabel("QG3 (MVAr)")
    plt.tight_layout()
    plt.savefig("reactive_power.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()