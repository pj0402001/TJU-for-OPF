# filename: wb5_roi_SEPs_UEP_refined.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

import pyomo.environ as pyo

# =============== 参数 ===============
SEED_RANDOM = 1
np.random.seed(SEED_RANDOM)

# 三个 ROI 的中心（单位 MW）：
ROI_SEPs1_CENTER   = (190.0, 240.0)   # 左侧 SEPs
ROI_SEPs2_CENTER   = (567.0, 169.0)   # 右侧 SEPs2
ROI_UEP_CENTER = (567.94, 159.3)   # 右侧 UEP

# 初始半宽；右侧两点相近，半宽要小些以避免互相“抢解”
ROI_HALF_SEPs1     = 10.0
ROI_HALF_SEPs2     = 1.0
ROI_HALF_UEP   = 1.0

# 多初值与缩盒
MULTISTARTS = 28                  # 每层多初值次数
ZOOM_SCALES = [1.0, 0.55, 0.35]   # 逐层缩盒比例
DEDUP_TOL   = 4.0                 # 解去重距离阈值（MW）
IPOPT_MAXITER = 4000

# =============== 5 节点网络数据 ===============
baseMVA = 100.0
bus = np.array([
    [1, 3,   0,   0, 0, 0, 1, 1.0,   0.0, 345, 1, 1.13, 0.87],
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
    m.BUS = pyo.Set(initialize=buses); m.GEN = pyo.Set(initialize=gens); m.LINE = pyo.Set(initialize=lines)
    m.Pd = pyo.Param(m.BUS, initialize=Pd); m.Qd = pyo.Param(m.BUS, initialize=Qd)
    m.Vmax = pyo.Param(m.BUS, initialize=Vmax); m.Vmin = pyo.Param(m.BUS, initialize=Vmin)
    m.Pmax = pyo.Param(m.GEN, initialize=Pmax); m.Pmin = pyo.Param(m.GEN, initialize=Pmin)
    m.Qmax = pyo.Param(m.GEN, initialize=Qmax); m.Qmin = pyo.Param(m.GEN, initialize=Qmin)
    m.fbus = pyo.Param(m.LINE, initialize=fbus, within=m.BUS)
    m.tbus = pyo.Param(m.LINE, initialize=tbus, within=m.BUS)
    m.g = pyo.Param(m.LINE, initialize=g_series); m.b = pyo.Param(m.LINE, initialize=b_series)
    m.bc = pyo.Param(m.LINE, initialize=b_total); m.rateA = pyo.Param(m.LINE, initialize=rateA)

    m.Vm = pyo.Var(m.BUS); m.Va = pyo.Var(m.BUS)
    m.Pg = pyo.Var(m.GEN); m.Qg = pyo.Var(m.GEN)

    def ref_rule(_m, i): return _m.Va[i]==0.0 if i==swing_bus else pyo.Constraint.Skip
    m.ref = pyo.Constraint(m.BUS, rule=ref_rule)

    for i in m.BUS:
        m.Vm[i].setlb(pyo.value(m.Vmin[i])); m.Vm[i].setub(pyo.value(m.Vmax[i]))
    for g in m.GEN:
        m.Pg[g].setlb(pyo.value(m.Pmin[g])); m.Pg[g].setub(pyo.value(m.Pmax[g]))
        m.Qg[g].setlb(pyo.value(m.Qmin[g])); m.Qg[g].setub(pyo.value(m.Qmax[g]))

    if pg_box is not None:
        (pg1_lo, pg1_hi), (pg2_lo, pg2_hi) = pg_box
        m.Pg[1].setlb(pg1_lo/baseMVA); m.Pg[1].setub(pg1_hi/baseMVA)
        m.Pg[2].setlb(pg2_lo/baseMVA); m.Pg[2].setub(pg2_hi/baseMVA)

    expr = sum(m.Pg[g]**2 for g in m.GEN)
    m.OBJ = pyo.Objective(expr=expr, sense=pyo.minimize if sense=='min' else pyo.maximize)

    Gmat, Bmat = G, B
    def Pbal(_m,i):
        Pi=0.0
        for j in _m.BUS:
            gij=Gmat[i-1,j-1]; bij=Bmat[i-1,j-1]
            Pi += _m.Vm[i]*_m.Vm[j]*(gij*pyo.cos(_m.Va[i]-_m.Va[j]) + bij*pyo.sin(_m.Va[i]-_m.Va[j]))
        return Pi == sum(_m.Pg[g] for g in _m.GEN if gen_bus[g]==i) - _m.Pd[i]
    def Qbal(_m,i):
        Qi=0.0
        for j in _m.BUS:
            gij=Gmat[i-1,j-1]; bij=Bmat[i-1,j-1]
            Qi += _m.Vm[i]*_m.Vm[j]*(gij*pyo.sin(_m.Va[i]-_m.Va[j]) - bij*pyo.cos(_m.Va[i]-_m.Va[j]))
        return Qi == sum(_m.Qg[g] for g in _m.GEN if gen_bus[g]==i) - _m.Qd[i]
    m.Pbal = pyo.Constraint(m.BUS, rule=Pbal); m.Qbal = pyo.Constraint(m.BUS, rule=Qbal)

    m.Sf = pyo.ConstraintList(); m.St = pyo.ConstraintList()
    for l in m.LINE:
        i=fbus[l]; j=tbus[l]; g=g_series[l]; b=b_series[l]; bc=b_total[l]
        def Pft(_m,i=i,j=j,g=g,b=b):
            return g*_m.Vm[i]**2 - g*_m.Vm[i]*_m.Vm[j]*pyo.cos(_m.Va[i]-_m.Va[j]) - b*_m.Vm[i]*_m.Vm[j]*pyo.sin(_m.Va[i]-_m.Va[j])
        def Qft(_m,i=i,j=j,g=g,b=b,bc=bc):
            return -(b + bc/2.0)*_m.Vm[i]**2 + b*_m.Vm[i]*_m.Vm[j]*pyo.cos(_m.Va[i]-_m.Va[j]) - g*_m.Vm[i]*_m.Vm[j]*pyo.sin(_m.Va[i]-_m.Va[j])
        def Ptf(_m,i=i,j=j,g=g,b=b):
            return g*_m.Vm[j]**2 - g*_m.Vm[i]*_m.Vm[j]*pyo.cos(_m.Va[i]-_m.Va[j]) + b*_m.Vm[i]*_m.Vm[j]*pyo.sin(_m.Va[i]-_m.Va[j])
        def Qtf(_m,i=i,j=j,g=g,b=b,bc=bc):
            return -(b + bc/2.0)*_m.Vm[j]**2 + b*_m.Vm[i]*_m.Vm[j]*pyo.cos(_m.Va[i]-_m.Va[j]) + g*_m.Vm[i]*_m.Vm[j]*pyo.sin(_m.Va[i]-_m.Va[j])
        m.Sf.add(Pft(m)**2 + Qft(m)**2 <= (rateA[l])**2)
        m.St.add(Ptf(m)**2 + Qtf(m)**2 <= (rateA[l])**2)

    # 初值
    for i in m.BUS:
        if init_vmva is not None:
            Vm0, Va0 = init_vmva.get(i, (1.0, 0.0))
            m.Vm[i].value = Vm0; m.Va[i].value = Va0
        else:
            m.Vm[i].value = 1.0; m.Va[i].value = 0.0
    if init_pg is not None:
        m.Pg[1].value = init_pg[0]/baseMVA
        m.Pg[2].value = init_pg[1]/baseMVA
    return m

def _run_ipopt(m: pyo.ConcreteModel) -> Dict:
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol']=1e-8
    solver.options['acceptable_tol']=1e-6
    solver.options['max_iter']=IPOPT_MAXITER
    res = solver.solve(m, tee=False)
    term=res.solver.termination_condition; stat=res.solver.status
    ok=(stat in (pyo.SolverStatus.ok, pyo.SolverStatus.warning)) and \
       (term in (pyo.TerminationCondition.optimal,
                 pyo.TerminationCondition.locallyOptimal,
                 pyo.TerminationCondition.feasible))
    if not ok: return {'ok':False}
    pg1=float(pyo.value(m.Pg[1])*baseMVA); pg2=float(pyo.value(m.Pg[2])*baseMVA)
    obj=float(pyo.value(sum(m.Pg[g]**2 for g in m.GEN)))
    return {'ok':True, 'PG1':pg1, 'PG2':pg2, 'obj':obj}

def solve_opf_zoom(center: Tuple[float,float], halfwidth: float, sense='min',
                   n_starts: int=MULTISTARTS, scales: List[float]=ZOOM_SCALES) -> Dict:
    cx, cy = center
    best=None
    for s in scales:
        hw = halfwidth * s
        box = ((cx-hw, cx+hw), (cy-hw, cy+hw))
        sols=[]
        # 构造多初值：随机 + 角点 + 中心
        for k in range(max(6, n_starts)):
            if k < 4:
                # 四角
                dx = hw if (k%2)==0 else -hw
                dy = hw if (k//2)==0 else -hw
                init_pg = (cx+dx, cy+dy)
            elif k == 4:
                init_pg = (cx, cy)  # 中心
            else:
                init_pg = (np.random.uniform(box[0][0], box[0][1]),
                           np.random.uniform(box[1][0], box[1][1]))
            init_vmva = {i: (1.0+0.02*np.random.randn(), 0.05*np.random.randn()) for i in buses}
            init_vmva[swing_bus]=(1.0,0.0)
            m = build_opf_model(pg_box=box, sense=sense, init_pg=init_pg, init_vmva=init_vmva)
            r = _run_ipopt(m)
            if r['ok']:
                sols.append({'PG1':r['PG1'], 'PG2':r['PG2'], 'obj':r['obj']})
        # 去重
        uniq=[]
        for s0 in sols:
            if len(uniq)==0:
                uniq.append(s0); continue
            d = np.min([np.hypot(s0['PG1']-u['PG1'], s0['PG2']-u['PG2']) for u in uniq])
            if d > DEDUP_TOL:
                uniq.append(s0)
        if len(uniq)==0: continue
        # 选优
        cand = min(uniq, key=lambda z:z['obj']) if sense=='min' else max(uniq, key=lambda z:z['obj'])
        best = cand if (best is None or
                        (sense=='min' and cand['obj']<best['obj']-1e-9) or
                        (sense=='max' and cand['obj']>best['obj']+1e-9)) else best
        # 以当前最佳为中心继续缩盒
        cx, cy = best['PG1'], best['PG2']
    return {'best':best}

# =============== CSV（仅用于可视化散点） ===============
def solve_opf_multistart_global_ref() -> Tuple[float,float]:
    best=None
    for _ in range(4):
        m=build_opf_model(pg_box=None, sense='min')
        r=_run_ipopt(m)
        if r.get('ok'):
            if best is None or r['obj']<best[0]:
                best=(r['obj'], r['PG1'], r['PG2'])
    if best is None: raise RuntimeError("OPF reference failed.")
    return best[1], best[2]

def load_feasible_csv_autoorient(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols=[]
    for c in raw.columns:
        s=pd.to_numeric(raw[c], errors='coerce')
        if s.notna().mean()>0.9: raw[c]=s; cols.append(c)
    if len(cols)<2: raise RuntimeError("CSV 至少两列（两台机组 PG）")
    A=raw[cols[0]].to_numpy(float); B=raw[cols[1]].to_numpy(float)
    r1,r2=solve_opf_multistart_global_ref()
    dA=np.min(np.hypot(A-r1, B-r2)); dB=np.min(np.hypot(B-r1, A-r2))
    if dB<dA: df=pd.DataFrame({'PG1':B,'PG2':A})
    else:     df=pd.DataFrame({'PG1':A,'PG2':B})
    return df.dropna().drop_duplicates().reset_index(drop=True)

# =============== 主流程 ===============
def main():
    # 1) 读可行域散点，仅用于背景展示
    try:
        df = load_feasible_csv_autoorient("5节点数据.csv")
        xy = df[['PG1','PG2']].to_numpy(float)
    except Exception as e:
        print(f"[warn] 无法读取 5节点数据.csv：{e}; 将不显示散点。")
        xy = None

    # 2) 在 ROI 内用缩盒+多初值求解
    SEPs1  = solve_opf_zoom(ROI_SEPs1_CENTER, ROI_HALF_SEPs1, sense='min')
    SEPs2  = solve_opf_zoom(ROI_SEPs2_CENTER, ROI_HALF_SEPs2, sense='min')
    UEP= solve_opf_zoom(ROI_UEP_CENTER, ROI_HALF_UEP, sense='max')

    print("[SEPs1]", SEPs1['best'])
    print("[SEPs2]", SEPs2['best'])
    print("[UEP]", UEP['best'])

    # 3) 画图（不显示 ROI 轮廓）
    plt.figure(figsize=(8,6))
    if xy is not None:
        plt.scatter(xy[:,0], xy[:,1], s=5, c='tab:blue', alpha=0.6, label='Feasible Region')
    if SEPs1['best'] is not None:
        plt.scatter([SEPs1['best']['PG1']], [SEPs1['best']['PG2']],
                    s=150, marker='*', c='crimson', edgecolors='k', linewidths=0.6,
                    label='SEPs')
    if SEPs2['best'] is not None:
        plt.scatter([SEPs2['best']['PG1']], [SEPs2['best']['PG2']],
                    s=150, marker='*', c='crimson', edgecolors='k', linewidths=0.6,
                    label='SEPs ' if SEPs1['best'] is None else None)
    if UEP['best'] is not None:
        plt.scatter([UEP['best']['PG1']], [UEP['best']['PG2']],
                    s=170, marker='*', c='limegreen', edgecolors='k', linewidths=0.6,
                    label='UEP')
    plt.xlabel("PG1 (MW)")
    plt.ylabel("PG2 (MW)")
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()