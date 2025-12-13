import numpy as np
import sqlite3
import math
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# =========================
# LMBM3 (3-node case) 数据
# =========================
baseMVA = 100.0

# bus: [bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va(deg), baseKV, zone, Vmax, Vmin]
bus = np.array([
    [1, 3, 110, 40, 0, 0, 1, 1.069, 0.000, 345, 1, 1.10, 0.90],
    [2, 2, 110, 40, 0, 0, 1, 1.028, 9.916, 345, 1, 1.10, 0.90],
    [3, 2, 95, 50, 0, 0, 1, 1.001, -13.561, 345, 1, 1.10, 0.90],
], dtype=float)

# gen: [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin]
gen = np.array([
    [1, 131.09, 17.02, 10000, -1000, 1.069, 100, 1, 10000, 0],
    [2, 185.93, -3.50, 1000, -1000, 1.028, 100, 1, 10000, 0],
    [3, 0.00, 0.06, 1000, -1000, 1.001, 100, 1, 0, 0],
], dtype=float)

# branch: [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]
branch = np.array([
    [1, 3, 0.065, 0.620, 0.450, 9999, 9999, 9999, 0, 0, 1, -360, 360],
    [3, 2, 0.025, 0.750, 0.700, 186, 9999, 9999, 0, 0, 1, -360, 360],
    [1, 2, 0.042, 0.900, 0.300, 9999, 9999, 9999, 0, 0, 1, -360, 360],
], dtype=float)

LOAD_FACTOR = 2.4  # Pd/Qd 放大进入潮流

# ================= 数据库 =================
def init_db():
    if os.path.exists('opf_results_ipopt.db'):
        os.remove('opf_results_ipopt.db')
    conn = sqlite3.connect('opf_results_ipopt.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE feasible_points
                      (Pg1_MW REAL, V1 REAL, V2 REAL, loss REAL, epoch_count INTEGER, compute_time REAL)''')
    conn.commit()
    return conn

# ============== 优先 Appsi-Ipopt 的 Pyomo 求解（含初值钳制） ==============
def solve_with_ipopt_3bus(init, Pg1_pu, P_load_pu, Q_load_pu, R_pu, X_pu,
                          Vmin, Vmax, Qg_min, Qg_max,
                          ipopt_max_iter=300, ipopt_tol=1e-8, silent=True):
    try:
        model = pyo.ConcreteModel()
        model.V1 = pyo.Var(bounds=(Vmin, Vmax), initialize=init['V1'])
        model.V2 = pyo.Var(bounds=(Vmin, Vmax), initialize=init['V2'])
        model.theta2 = pyo.Var(bounds=(-math.pi/2, math.pi/2), initialize=init['theta2'])
        model.Qg1 = pyo.Var(bounds=(Qg_min, Qg_max), initialize=init['Qg1'])

        # 常量
        Y = 1 / complex(R_pu, X_pu)
        G = float(Y.real)
        B = float(Y.imag)

        model.c1 = pyo.Constraint(expr=G * model.V1**2 - model.V1 * model.V2 * (G * pyo.cos(-model.theta2) + B * pyo.sin(-model.theta2)) - Pg1_pu == 0)
        model.c2 = pyo.Constraint(expr=-B * model.V1**2 - model.V1 * model.V2 * (G * pyo.sin(-model.theta2) - B * pyo.cos(-model.theta2)) - model.Qg1 == 0)
        model.c3 = pyo.Constraint(expr=G * model.V2**2 - model.V1 * model.V2 * (G * pyo.cos(-model.theta2) - B * pyo.sin(-model.theta2)) + P_load_pu == 0)
        model.c4 = pyo.Constraint(expr=-B * model.V2**2 + model.V1 * model.V2 * (G * pyo.sin(-model.theta2) + B * pyo.cos(-model.theta2)) + Q_load_pu == 0)

        # 可行性问题：目标置零
        model.obj = pyo.Objective(expr=0.0, sense=pyo.minimize)

        # ====== 使用外部 ipopt 可执行文件 ======
        solver = SolverFactory('ipopt')
        if (solver is None) or (not solver.available(False)):
            if not silent:
                print("[外部Ipopt] 不可用（未安装或不在 PATH）")
            return False, None, None

        solver.options['print_level'] = 0 if silent else 5
        solver.options['tol'] = ipopt_tol
        solver.options['acceptable_tol'] = max(ipopt_tol * 10, 1e-6)
        solver.options['max_iter'] = ipopt_max_iter
        solver.options['mu_strategy'] = "adaptive"
        solver.options['linear_solver'] = "mumps"

        res = solver.solve(model, tee=not silent)
        ok = (res.solver.termination_condition == pyo.TerminationCondition.optimal)
        if ok:
            sol = {
                'V1': pyo.value(model.V1),
                'V2': pyo.value(model.V2),
                'theta2': pyo.value(model.theta2),
                'Qg1': pyo.value(model.Qg1),
            }
            return True, sol, res
        else:
            return False, None, res

    except Exception as e:
        print(f"[IPOPT异常] {e}")
        return False, None, None

# ============== 主流程：仅使用 IPOPT ==============
def compute_feasible_region_opf():
    conn = init_db()
    S_base = baseMVA
    P_load_pu = bus[1, 2] / S_base  # 第二个节点的负荷
    Q_load_pu = bus[1, 3] / S_base  # 第二个节点的无功负荷
    R_pu = branch[0, 2]
    X_pu = branch[0, 3]

    num_points = 100
    Pg1_points_MW = np.linspace(440, 461.5, num_points)

    with open("optimization_log.csv", "w") as log_file:
        log_file.write("Pg1_MW,Start_V1,Start_V2,Best_V1,Best_V2,Final_Loss,Epochs,Compute_Time,IpoptOk\n")

        for Pg1_MW in tqdm(Pg1_points_MW, desc="IPOPT 计算"):
            Pg1_pu = Pg1_MW / S_base
            init = {'V1': bus[0, 7], 'V2': bus[1, 7], 'theta2': 0.0, 'Qg1': 0.0}  # 初始值设定

            start_time = time.time()

            ok, sol, res = solve_with_ipopt_3bus(
                init=init, Pg1_pu=Pg1_pu, P_load_pu=P_load_pu, Q_load_pu=Q_load_pu,
                R_pu=R_pu, X_pu=X_pu, Vmin=float('-inf'), Vmax=float('inf'),
                Qg_min=float('-inf'), Qg_max=float('inf'),
                ipopt_max_iter=500, ipopt_tol=1e-8, silent=True
            )

            end_time = time.time()
            compute_time = end_time - start_time

            # 记录结果
            if ok and sol is not None:
                V1_val, V2_val = sol['V1'], sol['V2']
                log_line = f"{Pg1_MW},{init['V1']},{init['V2']},{V1_val},{V2_val},{0.0},{0},{compute_time:.2f},{1}"
                conn.execute(
                    "INSERT INTO feasible_points (Pg1_MW, V1, V2, loss, epoch_count, compute_time) VALUES (?, ?, ?, ?, ?, ?)",
                    (Pg1_MW, V1_val, V2_val, 0.0, 0, compute_time)
                )
                print(f"[OK] Pg1={Pg1_MW:.1f}MW 收敛: V1={V1_val:.4f}, V2={V2_val:.4f}, time={compute_time:.2f}s")
            else:
                log_line = f"{Pg1_MW},{init['V1']},{init['V2']},{0},{0},{float('inf')},{0},{compute_time:.2f},{0}"

            log_file.write(log_line + "\n")
            conn.commit()

    conn.close()

# ============== 可视化 ==============
def visualize_results():
    if not os.path.exists('opf_results_ipopt.db'):
        print("数据库未找到，请先运行计算过程")
        return
    conn = sqlite3.connect('opf_results_ipopt.db')
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM feasible_points")
        feasible_data = c.fetchall()
        conn.close()

        if not feasible_data:
            print("未找到任何结果！请查看日志文件分析原因")
            return

        fig, ax1 = plt.subplots(figsize=(8, 6))
        Pg1 = [row[0] for row in feasible_data]
        V1 = [row[1] for row in feasible_data]
        V2 = [row[2] for row in feasible_data]
        compute_times = [row[5] for row in feasible_data]

        scatter = ax1.scatter(V1, V2, c=Pg1, cmap="viridis", s=20, alpha=0.6)
        fig.colorbar(scatter, ax=ax1, label="Pg1 (MW)")
        ax1.set_xlabel("V1 (p.u.)")
        ax1.set_ylabel("V2 (p.u.)")
        ax1.grid(True, linestyle='--', alpha=0.1)
        ax1.set_title("收敛解电压分布")
        plt.tight_layout()
        plt.savefig("ipopt_results.png", dpi=300)
        plt.show()

    except sqlite3.OperationalError as e:
        print(f"数据库查询错误: {e}")
        conn.close()

# ============== 入口 ==============
if __name__ == "__main__":
    if os.path.exists('optimization_log.csv'):
        os.remove('optimization_log.csv')
    compute_feasible_region_opf()
    visualize_results()