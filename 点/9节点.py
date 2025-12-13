import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import pyomo.environ as pyo

# =============== 参数 ===============
SEED_RANDOM = 1
np.random.seed(SEED_RANDOM)

# 9个节点网络数据（根据实际情况调整）
baseMVA = 100.0
bus = np.array([
    [1, 3,   0,   0, 0, 0, 1, 1.0,   0.0, 345, 1, 1.13, 0.87],
    [2, 1, 130,  20, 0, 0, 1, 1.0, -10.0, 345, 1, 1.13, 0.87],
    [3, 1, 130,  20, 0, 0, 1, 1.0, -20.0, 345, 1, 1.13, 0.87],
    [4, 1,  65,  10, 0, 0, 1, 1.0,-135.0, 345, 1, 1.13, 0.87],
    [5, 2,   0,   0, 0, 0, 1, 1.0,-140.0, 345, 1, 1.13, 0.87],
    [6, 1,   0,   0, 0, 0, 1, 1.0,-150.0, 345, 1, 1.13, 0.87],
    [7, 1,  30,  10, 0, 0, 1, 1.0,-160.0, 345, 1, 1.13, 0.87],
    [8, 2,  50,   0, 0, 0, 1, 1.0,-170.0, 345, 1, 1.13, 0.87],
    [9, 1,  20,  15, 0, 0, 1, 1.0,-180.0, 345, 1, 1.13, 0.87],
], dtype=float)
gen = np.array([
    [1, 500,  50, 1800,  -30, 1.0, 100, 1, 5000,   0],
    [5,   0,   0, 1800,  -30, 1.0, 100, 1, 5000,   0],
    [6,   0,   0, 1800,  -30, 1.0, 100, 1, 5000,   0],
], dtype=float)
branch = np.array([
    [1, 2, 0.04, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [1, 3, 0.05, 0.10, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [2, 4, 0.55, 0.90, 0.45, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [3, 5, 0.55, 0.90, 0.45, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [4, 5, 0.06, 0.10, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [2, 3, 0.07, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [5, 6, 0.05, 0.08, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [3, 7, 0.04, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [6, 8, 0.05, 0.10, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
], dtype=float)

def series_y(r, x):
    z = complex(r, x)
    y = 1.0 / z
    return y.real, y.imag

# 构建电网数据
nb, nl, ng = bus.shape[0], branch.shape[0], gen.shape[0]
buses = list(range(1, nb + 1))
lines = list(range(1, nl + 1))
gens = list(range(1, ng + 1))

gen_bus = {i + 1: int(gen[i, 0]) for i in range(ng)}
Pmax = {i + 1: gen[i, 8] / baseMVA for i in range(ng)}
Pmin = {i + 1: gen[i, 9] / baseMVA for i in range(ng)}
Qmax = {i + 1: gen[i, 3] / baseMVA for i in range(ng)}
Qmin = {i + 1: gen[i, 4] / baseMVA for i in range(ng)}

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

g_series = {l + 1: series_y(r[l + 1], x[l + 1])[0] for l in range(nl)}
b_series = {l + 1: series_y(r[l + 1], x[l + 1])[1] for l in range(nl)}

# 构造 Ybus
Ybus = np.zeros((nb, nb), dtype=complex)
for l in lines:
    i = fbus[l] - 1; j = tbus[l] - 1
    y = 1.0 / complex(r[l], x[l])
    bsh = 1j * (b_total[l] / 2.0)
    Ybus[i, i] += y + bsh; Ybus[j, j] += y + bsh
    Ybus[i, j] -= y; Ybus[j, i] -= y
G = Ybus.real; B = Ybus.imag
swing_bus = 1

# =============== OPF 模型 ===============
def build_opf_model(pg_box=None, sense='min', init_pg=None, init_vmva=None):
    m = pyo.ConcreteModel()
    m.BUS = pyo.Set(initialize=buses)
    m.GEN = pyo.Set(initialize=gens)
    m.LINE = pyo.Set(initialize=lines)
    m.Pd = pyo.Param(m.BUS, initialize=Pd)
    m.Qd = pyo.Param(m.BUS, initialize=Qd)
    m.Vmax = pyo.Param(m.BUS, initialize=Vmax)
    m.Vmin = pyo.Param(m.BUS, initialize=Vmin)
    m.Pmax = pyo.Param(m.GEN, initialize=Pmax)
    m.Pmin = pyo.Param(m.GEN, initialize=Pmin)
    m.Qmax = pyo.Param(m.GEN, initialize=Qmax)
    m.Qmin = pyo.Param(m.GEN, initialize=Qmin)
    m.fbus = pyo.Param(m.LINE, initialize=fbus, within=m.BUS)
    m.tbus = pyo.Param(m.LINE, initialize=tbus, within=m.BUS)
    m.g = pyo.Param(m.LINE, initialize=g_series)
    m.b = pyo.Param(m.LINE, initialize=b_series)
    m.bc = pyo.Param(m.LINE, initialize=b_total)
    m.rateA = pyo.Param(m.LINE, initialize=rateA)

    m.Vm = pyo.Var(m.BUS)
    m.Va = pyo.Var(m.BUS)
    m.Pg = pyo.Var(m.GEN)
    m.Qg = pyo.Var(m.GEN)

    def ref_rule(_m, i):
        return _m.Va[i] == 0.0 if i == swing_bus else pyo.Constraint.Skip
    m.ref = pyo.Constraint(m.BUS, rule=ref_rule)

    for i in m.BUS:
        m.Vm[i].setlb(pyo.value(m.Vmin[i]))
        m.Vm[i].setub(pyo.value(m.Vmax[i]))
    for g in m.GEN:
        m.Pg[g].setlb(pyo.value(m.Pmin[g]))
        m.Pg[g].setub(pyo.value(m.Pmax[g]))
        m.Qg[g].setlb(pyo.value(m.Qmin[g]))
        m.Qg[g].setub(pyo.value(m.Qmax[g]))

    if pg_box is not None:
        (pg1_lo, pg1_hi), (pg2_lo, pg2_hi) = pg_box
        m.Pg[1].setlb(pg1_lo / baseMVA)
        m.Pg[1].setub(pg1_hi / baseMVA)
        m.Pg[2].setlb(pg2_lo / baseMVA)
        m.Pg[2].setub(pg2_hi / baseMVA)

    expr = sum(m.Pg[g] ** 2 for g in m.GEN)
    m.OBJ = pyo.Objective(expr=expr, sense=pyo.minimize if sense == 'min' else pyo.maximize)

    Gmat, Bmat = G, B

    def Pbal(_m, i):
        Pi = 0.0
        for j in _m.BUS:
            gij = Gmat[i - 1, j - 1]
            bij = Bmat[i - 1, j - 1]
            Pi += _m.Vm[i] * _m.Vm[j] * (gij * pyo.cos(_m.Va[i] - _m.Va[j]) + bij * pyo.sin(_m.Va[i] - _m.Va[j]))
        return Pi == sum(_m.Pg[g] for g in _m.GEN if gen_bus[g] == i) - _m.Pd[i]

    def Qbal(_m, i):
        Qi = 0.0
        for j in _m.BUS:
            gij = Gmat[i - 1, j - 1]
            bij = Bmat[i - 1, j - 1]
            Qi += _m.Vm[i] * _m.Vm[j] * (gij * pyo.sin(_m.Va[i] - _m.Va[j]) - bij * pyo.cos(_m.Va[i] - _m.Va[j]))
        return Qi == sum(_m.Qg[g] for g in _m.GEN if gen_bus[g] == i) - _m.Qd[i]

    m.Pbal = pyo.Constraint(m.BUS, rule=Pbal)
    m.Qbal = pyo.Constraint(m.BUS, rule=Qbal)

    m.Sf = pyo.ConstraintList()
    m.St = pyo.ConstraintList()
    for l in m.LINE:
        i = fbus[l]
        j = tbus[l]
        g = g_series[l]
        b = b_series[l]
        bc = b_total[l]

        def Pft(_m, i=i, j=j, g=g, b=b):
            return g * _m.Vm[i] ** 2 - g * _m.Vm[i] * _m.Vm[j] * pyo.cos(_m.Va[i] - _m.Va[j]) - b * _m.Vm[i] * _m.Vm[j] * pyo.sin(_m.Va[i] - _m.Va[j])

        def Qft(_m, i=i, j=j, g=g, b=b, bc=bc):
            return -(b + bc / 2.0) * _m.Vm[i] ** 2 + b * _m.Vm[i] * _m.Vm[j] * pyo.cos(_m.Va[i] - _m.Va[j]) - g * _m.Vm[i] * _m.Vm[j] * pyo.sin(_m.Va[i] - _m.Va[j])

        def Ptf(_m, i=i, j=j, g=g, b=b):
            return g * _m.Vm[j] ** 2 - g * _m.Vm[i] * _m.Vm[j] * pyo.cos(_m.Va[i] - _m.Va[j]) + b * _m.Vm[i] * _m.Vm[j] * pyo.sin(_m.Va[i] - _m.Va[j])

        def Qtf(_m, i=i, j=j, g=g, b=b, bc=bc):
            return -(b + bc / 2.0) * _m.Vm[j] ** 2 + b * _m.Vm[i] * _m.Vm[j] * pyo.cos(_m.Va[i] - _m.Va[j]) + g * _m.Vm[i] * _m.Vm[j] * pyo.sin(_m.Va[i] - _m.Va[j])

        m.Sf.add(Pft(m) ** 2 + Qft(m) ** 2 <= (rateA[l]) ** 2)
        m.St.add(Ptf(m) ** 2 + Qtf(m) ** 2 <= (rateA[l]) ** 2)

    # 初值设定
    for i in m.BUS:
        if init_vmva is not None:
            Vm0, Va0 = init_vmva.get(i, (1.0, 0.0))
            m.Vm[i].value = Vm0
            m.Va[i].value = Va0
        else:
            m.Vm[i].value = 1.0
            m.Va[i].value = 0.0
    if init_pg is not None:
        m.Pg[1].value = init_pg[0] / baseMVA
        m.Pg[2].value = init_pg[1] / baseMVA
    return m

def _run_ipopt(m: pyo.ConcreteModel) -> Dict:
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-8
    solver.options['acceptable_tol'] = 1e-6
    solver.options['max_iter'] = 4000
    res = solver.solve(m, tee=False)
    term = res.solver.termination_condition
    stat = res.solver.status
    ok = (stat in (pyo.SolverStatus.ok, pyo.SolverStatus.warning)) and \
         (term in (pyo.TerminationCondition.optimal,
                    pyo.TerminationCondition.locallyOptimal,
                    pyo.TerminationCondition.feasible))
    if not ok: return {'ok': False}
    pg1 = float(pyo.value(m.Pg[1]) * baseMVA)
    pg2 = float(pyo.value(m.Pg[2]) * baseMVA)
    obj = float(pyo.value(sum(m.Pg[g] ** 2 for g in m.GEN)))
    return {'ok': True, 'PG1': pg1, 'PG2': pg2, 'obj': obj}

def load_feasible_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = ['p1_mw', 'p2_mw']  # 使用对应的列名
    if not all(col in raw.columns for col in cols):
        raise RuntimeError("CSV文件不包含必要的列：p1_mw, p2_mw")
    return raw[cols].dropna().drop_duplicates().reset_index(drop=True)

# =============== 主流程 ===============
def main():
    # 读取9节点数据的可行域散点
    try:
        df = load_feasible_csv("ac_opf_9results.csv")
        xy = df[['p1_mw', 'p2_mw']].to_numpy(float)

        # 设定LOS和Saddle Point数据
        los_points = [
            {'PG1': 11.6353, 'PG2': 125.41},
            {'PG1': 10.53, 'PG2': 63.34},
            {'PG1': 143.889, 'PG2': 37.15},
            {'PG1': 142.3557, 'PG2': 9.9999}
        ]
        saddle = {'PG1': 143.559, 'PG2': 14.83}  # 示例Saddle Point

    except Exception as e:
        print(f"[warn] 无法读取 9节点数据文件：{e}; 将不显示散点。")
        xy = None
        los_points = []
        saddle = None

    # 开始绘图
    plt.figure(figsize=(8, 6))
    if xy is not None:
        plt.scatter(xy[:, 0], xy[:, 1], s=5, c='tab:blue', alpha=0.6, label='Feasible Region')

    # 绘制LOS点
    for los in los_points:
        plt.scatter([los['PG1']], [los['PG2']],
                    s=150, marker='*', c='crimson', edgecolors='k', linewidths=1.0,
                    label='SEPs' if los == los_points[0] else None)

    # 绘制Saddle Point
    if saddle is not None:
        plt.scatter([saddle['PG1']], [saddle['PG2']],
                    s=170, marker='*', c='limegreen', edgecolors='k', linewidths=0.6,
                    label='UEPs')

    plt.xlabel("PG1 (MW)")
    plt.ylabel("PG2 (MW)")
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()