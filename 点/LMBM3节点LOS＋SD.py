import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, List
import pyomo.environ as pyo

# =============== 参数 ===============
SEED_RANDOM = 1
np.random.seed(SEED_RANDOM)

# 已知的关键点位置（根据您提供的信息）
LOS1 = (198, 280)
LOS2 = (175, 330) 
SADDLE1 = (401.5, 110.0)
SADDLE2 = (115, 373)

# ROI 中心 - 修正为正确的已知点
ROI_LOS1_CENTER   = LOS1
ROI_LOS2_CENTER   = LOS2
ROI_SADDLE1_CENTER = SADDLE1
ROI_SADDLE2_CENTER = SADDLE2

# 初始半宽 - 调整到合适的范围
ROI_HALF_LOS1     = 30.0
ROI_HALF_LOS2     = 30.0
ROI_HALF_SADDLE1   = 30.0
ROI_HALF_SADDLE2   = 30.0

MULTISTARTS = 28
ZOOM_SCALES = [1.0, 0.55, 0.35, 0.2]  # 增加一个更小的缩放比例
DEDUP_TOL   = 4.0
IPOPT_MAXITER = 4000

# =============== 网络数据 ===============
baseMVA = 100.0
bus = np.array([
    [1, 3,   0,   0, 0, 0, 1, 1.0,   0.0, 345, 1, 1.13, 0.87],
    [2, 1, 130,  20, 0, 0, 1, 1.0, -10.0, 345, 1, 1.13, 0.87],
    [3, 1,  65,  10, 0, 0, 1, 1.0, -20.0, 345, 1, 1.13, 0.87],
], dtype=float)

gen = np.array([
    [1, 500,  50, 1800,  -30, 1.0, 100, 1, 5000,   0],
    [2, 130,   0, 1800,  -30, 1.0, 100, 1, 5000,   0],
], dtype=float)

branch = np.array([
    [1, 2, 0.04, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [1, 3, 0.05, 0.10, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
    [2, 3, 0.07, 0.09, 0.00, 2500, 2500, 2500, 0, 0, 1, -360, 360],
], dtype=float)

# 处理 Ybus 和其他参数
def series_y(r, x):
    z = complex(r, x)
    y = 1.0 / z
    return y.real, y.imag

nb, nl, ng = bus.shape[0], branch.shape[0], gen.shape[0]
buses = list(range(1, nb+1))
lines = list(range(1, nl+1))
gens = list(range(1, ng+1))

gen_bus = {i+1: int(gen[i,0]) for i in range(ng)}
Pmax = {i+1: gen[i,8]/baseMVA for i in range(ng)}
Pmin = {i+1: gen[i,9]/baseMVA for i in range(ng)}
Qmax = {i+1: gen[i,3]/baseMVA for i in range(ng)}
Qmin = {i+1: gen[i,4]/baseMVA for i in range(ng)}

Pd = {int(bus[i,0]): bus[i,2]/baseMVA for i in range(nb)}
Qd = {int(bus[i,0]): bus[i,3]/baseMVA for i in range(nb)}
Vmax = {int(bus[i,0]): bus[i,11] for i in range(nb)}
Vmin = {int(bus[i,0]): bus[i,12] for i in range(nb)}

fbus = {l+1: int(branch[l,0]) for l in range(nl)}
tbus = {l+1: int(branch[l,1]) for l in range(nl)}
r = {l+1: branch[l,2] for l in range(nl)}
x = {l+1: branch[l,3] for l in range(nl)}
b_total = {l+1: branch[l,4] for l in range(nl)}
rateA = {l+1: branch[l,5]/baseMVA for l in range(nl)}

g_series = {l+1: series_y(r[l+1], x[l+1])[0] for l in range(nl)}
b_series = {l+1: series_y(r[l+1], x[l+1])[1] for l in range(nl)}

# 构造 Ybus
Ybus = np.zeros((nb, nb), dtype=complex)
for l in lines:
    i = fbus[l]-1; j = tbus[l]-1
    y = 1.0/complex(r[l], x[l])
    bsh = 1j*(b_total[l]/2.0)
    Ybus[i,i] += y + bsh; Ybus[j,j] += y + bsh
    Ybus[i,j] -= y; Ybus[j,i] -= y
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

    def ref_rule(m, i):
        return m.Va[i] == 0.0 if i == swing_bus else pyo.Constraint.Skip
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

    expr = sum(m.Pg[g]**2 for g in m.GEN)
    m.OBJ = pyo.Objective(expr=expr, sense=pyo.minimize if sense == 'min' else pyo.maximize)

    Gmat, Bmat = G, B

    def Pbal(m, i):
        Pi = 0.0
        for j in m.BUS:
            gij = Gmat[i-1,j-1]
            bij = Bmat[i-1,j-1]
            Pi += m.Vm[i] * m.Vm[j] * (gij * pyo.cos(m.Va[i] - m.Va[j]) + bij * pyo.sin(m.Va[i] - m.Va[j]))
        return Pi == sum(m.Pg[g] for g in m.GEN if gen_bus[g] == i) - m.Pd[i]

    def Qbal(m, i):
        Qi = 0.0
        for j in m.BUS:
            gij = Gmat[i-1,j-1]
            bij = Bmat[i-1,j-1]
            Qi += m.Vm[i] * m.Vm[j] * (gij * pyo.sin(m.Va[i] - m.Va[j]) - bij * pyo.cos(m.Va[i] - m.Va[j]))
        return Qi == sum(m.Qg[g] for g in m.GEN if gen_bus[g] == i) - m.Qd[i]

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

        def Pft(m, i=i, j=j, g=g, b=b):
            return g * m.Vm[i]**2 - g * m.Vm[i] * m.Vm[j] * pyo.cos(m.Va[i] - m.Va[j]) - b * m.Vm[i] * m.Vm[j] * pyo.sin(m.Va[i] - m.Va[j])

        def Qft(m, i=i, j=j, g=g, b=b):
            return -(b + bc / 2.0) * m.Vm[i]**2 + b * m.Vm[i] * m.Vm[j] * pyo.cos(m.Va[i] - m.Va[j]) - g * m.Vm[i] * m.Vm[j] * pyo.sin(m.Va[i] - m.Va[j])
        
        m.Sf.add(Pft(m)**2 + Qft(m)**2 <= (rateA[l])**2)

    # 初始化变量
    for i in m.BUS:
        if init_vmva is not None:
            Vm0, Va0 = init_vmva.get(i, (1.0, 0.0))
            m.Vm[i].value = Vm0
            m.Va[i].value = Va0
        else:
            m.Vm[i].value = 1.0
            m.Va[i].value = 0.0

    if init_pg is not None:
        for g_idx, pg_val in enumerate(init_pg, start=1):
            if g_idx in m.GEN:
                m.Pg[g_idx].value = pg_val / baseMVA

    return m

def _run_ipopt(m: pyo.ConcreteModel) -> Dict:
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-8
    solver.options['acceptable_tol'] = 1e-6
    solver.options['max_iter'] = IPOPT_MAXITER
    solver.options['print_level'] = 0  # 减少输出
    res = solver.solve(m, tee=False)
    term = res.solver.termination_condition
    stat = res.solver.status
    ok = (stat in (pyo.SolverStatus.ok, pyo.SolverStatus.warning)) and \
         (term in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal, pyo.TerminationCondition.feasible))
    
    if not ok: 
        return {'ok': False}
    
    # 获取所有发电机出力
    pg_results = {}
    for g in m.GEN:
        pg_results[g] = float(pyo.value(m.Pg[g])) * baseMVA
    
    # 获取所有发电机无功
    qg_results = {}
    for g in m.GEN:
        qg_results[g] = float(pyo.value(m.Qg[g])) * baseMVA
    
    obj = float(pyo.value(sum(m.Pg[g]**2 for g in m.GEN)))

    return {'ok': True, 'PG': pg_results, 'QG': qg_results, 'obj': obj}

def solve_opf_zoom(center: Tuple[float, float], halfwidth: float, sense='min') -> Dict:
    cx, cy = center
    best = None
    for s in ZOOM_SCALES:
        hw = halfwidth * s
        box = ((cx - hw, cx + hw), (cy - hw, cy + hw))
        sols = []
        
        for k in range(max(6, MULTISTARTS)):
            if k < 4:
                dx = hw if (k % 2) == 0 else -hw
                dy = hw if (k // 2) == 0 else -hw
                init_pg = (cx + dx, cy + dy)
            elif k == 4:
                init_pg = (cx, cy)
            else:
                init_pg = (np.random.uniform(box[0][0], box[0][1]),
                           np.random.uniform(box[1][0], box[1][1]))
            
            init_vmva = {i: (1.0 + 0.02 * np.random.randn(), 0.05 * np.random.randn()) for i in buses}
            init_vmva[swing_bus] = (1.0, 0.0)
            try:
                m = build_opf_model(pg_box=box, sense=sense, init_pg=init_pg, init_vmva=init_vmva)
                r = _run_ipopt(m)
                if r['ok']:
                    sols.append(r)
            except Exception as e:
                continue

        # 去重
        uniq = []
        for s0 in sols:
            if len(uniq) == 0:
                uniq.append(s0)
                continue
            # 使用PG1和PG2作为去重依据
            d = np.min([np.hypot(s0['PG'][1] - u['PG'][1], s0['PG'][2] - u['PG'][2]) for u in uniq])
            if d > DEDUP_TOL:
                uniq.append(s0)

        if len(uniq) == 0: 
            continue
        
        # 选优
        cand = min(uniq, key=lambda z: z['obj']) if sense == 'min' else max(uniq, key=lambda z: z['obj'])
        if best is None:
            best = cand
        else:
            if sense == 'min' and cand['obj'] < best['obj'] - 1e-9:
                best = cand
            elif sense == 'max' and cand['obj'] > best['obj'] + 1e-9:
                best = cand
        
        # 以当前最佳为中心继续缩盒
        cx, cy = best['PG'][1], best['PG'][2]
    
    return {'best': best}

def load_feasible_csv_autoorient(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = []
    for c in raw.columns:
        s = pd.to_numeric(raw[c], errors='coerce')
        if s.notna().mean() > 0.9:
            raw[c] = s
            cols.append(c)
    if len(cols) < 6:
        raise RuntimeError("CSV 至少需要六列（PG2, PG1, QG1, QG2, QG3, Vm1, Vm2, Vm3）")

    # 正确映射到数据 - 修正列名
    columns_mapping = {
        'PG1': 0,
        'PG2': 1,
        'QG1': 2,
        'QG2': 3,
        'QG3': 4,
        'Vm1': 5,
        'Vm2': 6,
        'Vm3': 7,
        'Va1_deg': 8,
        'Va2_deg': 9,
        'Va3_deg': 10,
        'Ploss_MW': 11,
        'load_factor': 12
    }
    
    # 创建包含所有所需列的数据框
    result_df = pd.DataFrame()
    for col_name, col_idx in columns_mapping.items():
        if col_idx < len(raw.columns):
            result_df[col_name] = raw.iloc[:, col_idx]
    
    return result_df.dropna().drop_duplicates().reset_index(drop=True)

# =============== 主流程 ===============
def main():
    # 1) 读取可行域散点，用于背景展示
    try:
        df = load_feasible_csv_autoorient("lmbm3_feasible_points_v2_optimized.csv")
        print(f"成功读取数据，共 {len(df)} 个点")
        print("数据列:", df.columns.tolist())
        
        # 提取PG1, PG2, QG1, QG2, QG3
        pg1 = df['PG1'].values
        pg2 = df['PG2'].values
        qg1 = df['QG1'].values
        qg2 = df['QG2'].values
        qg3 = df['QG3'].values
        
    except Exception as e:
        print(f"[warn] 无法读取数据：{e}; 将不显示散点。")
        pg1, pg2, qg1, qg2, qg3 = None, None, None, None, None

    # 2) 在 ROI 内用缩盒+多初值求解 - 修正为四个关键点
    print("正在搜索 LOS1...")
    los1 = solve_opf_zoom(ROI_LOS1_CENTER, ROI_HALF_LOS1, sense='min')
    print("正在搜索 LOS2...")
    los2 = solve_opf_zoom(ROI_LOS2_CENTER, ROI_HALF_LOS2, sense='min')
    print("正在搜索 Saddle1...")
    saddle1 = solve_opf_zoom(ROI_SADDLE1_CENTER, ROI_HALF_SADDLE1, sense='max')
    print("正在搜索 Saddle2...")
    saddle2 = solve_opf_zoom(ROI_SADDLE2_CENTER, ROI_HALF_SADDLE2, sense='max')

    print("\n=== 搜索结果 ===")
    if los1['best'] is not None:
        print(f"[LOS1] PG1={los1['best']['PG'][1]:.1f}, PG2={los1['best']['PG'][2]:.1f}, obj={los1['best']['obj']:.4f}")
    else:
        print("[LOS1] 未找到")
    
    if los2['best'] is not None:
        print(f"[LOS2] PG1={los2['best']['PG'][1]:.1f}, PG2={los2['best']['PG'][2]:.1f}, obj={los2['best']['obj']:.4f}")
    else:
        print("[LOS2] 未找到")
    
    if saddle1['best'] is not None:
        print(f"[Saddle1] PG1={saddle1['best']['PG'][1]:.1f}, PG2={saddle1['best']['PG'][2]:.1f}, obj={saddle1['best']['obj']:.4f}")
    else:
        print("[Saddle1] 未找到")
    
    if saddle2['best'] is not None:
        print(f"[Saddle2] PG1={saddle2['best']['PG'][1]:.1f}, PG2={saddle2['best']['PG'][2]:.1f}, obj={saddle2['best']['obj']:.4f}")
    else:
        print("[Saddle2] 未找到")

    # 3) 画 PG1-PG2 散点图
    plt.figure(figsize=(10, 8))
    if pg1 is not None and pg2 is not None:
        plt.scatter(pg1, pg2, s=5, c='tab:blue', alpha=0.6, label='Feasible Region')
    
    # 标记已知的关键点位置
    plt.scatter([LOS1[0], LOS2[0]], [LOS1[1], LOS2[1]], 
                s=200, marker='*', c='deeppink', linewidths=1.0, edgecolors='k',label='LOS')
    plt.scatter([SADDLE1[0], SADDLE2[0]], [SADDLE1[1], SADDLE2[1]], 
                s=200, marker='*', c='limegreen', linewidths=1.0,edgecolors='k', label='Saddle Point')
    
    # 标记找到的点
    colors = ['crimson', 'darkorange', 'limegreen', 'darkviolet']
    labels = ['LOS1 Found', 'LOS2 Found', 'Saddle1 Found', 'Saddle2 Found']
    results = [los1, los2, saddle1, saddle2]
    
    for i, (result, color, label) in enumerate(zip(results, colors, labels)):
        if result['best'] is not None:
            pg = result['best']['PG']
            plt.scatter([pg[1]], [pg[2]], s=150, marker='*', c=color, 
                       edgecolors='k', linewidths=0.6, label=label, zorder=5)

    plt.xlabel("PG1 (MW)  [Generator at bus 1]")
    plt.ylabel("PG2 (MW)  [Generator at bus 2]")
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) 画 QG1-QG2-QG3 三维散点图 - 修正为使用正确的QG数据
    if qg1 is not None and qg2 is not None and qg3 is not None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制可行域
        ax.scatter(qg1, qg2, qg3, s=5, c='tab:blue', alpha=0.6, label='Feasible Region')
        
        # 标记找到的点
        for i, (result, color, label) in enumerate(zip(results, colors, labels)):
            if result['best'] is not None:
                qg = result['best']['QG']
                # 注意：这里我们只有2台发电机，但数据中有3个节点的QG
                # 我们需要从OPF结果中获取QG值
                ax.scatter([qg[1]], [qg[2]], [0], s=150, marker='*', c=color, 
                          edgecolors='k', linewidths=0.6, label=label, zorder=5)

        ax.set_xlabel("QG1 (MVAR)")
        ax.set_ylabel("QG2 (MVAR)")
        ax.set_zlabel("QG3 (MVAR)")
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("无法绘制QG三维图：缺少QG数据")

if __name__ == "__main__":
    main()