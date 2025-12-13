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

# Pyomo
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 系统参数（2 节点） =====
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

# ================= 数据库 =================
def init_db():
    if os.path.exists('opf_results_ipopt.db'):
        os.remove('opf_results_ipopt.db')
    conn = sqlite3.connect('opf_results_ipopt.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE feasible_points
                    (Pg1_MW REAL, V1 REAL, V2 REAL, loss REAL, epoch_count INTEGER, compute_time REAL)''')
    cursor.execute('''CREATE TABLE partial_results
                    (Pg1_MW REAL, best_loss REAL, epoch_count INTEGER)''')
    conn.commit()
    return conn

# ============== 初始化（根据 Pg 轻微自适应） ==============
def initialize_variables(Pg1_pu):
    base_V1 = 0.965 - 0.003 * (Pg1_pu - 4.4)
    base_V2 = 1.02  - 0.015 * (Pg1_pu - 4.4)
    return (
        torch.tensor([base_V1], requires_grad=True),
        torch.tensor([base_V2], requires_grad=True),
        torch.tensor([0.01], requires_grad=True),
        torch.tensor([0.0],  requires_grad=True)
    )

# ============== 平滑边界罚：softplus ==============
def smooth_barrier(x, lower, upper, coef=0.05, delta=0.02):
    sp = torch.nn.functional.softplus
    lower_viol = sp((lower - x) / delta) * delta
    upper_viol = sp((x - upper) / delta) * delta
    return coef * torch.sum(lower_viol + upper_viol)

# ============== 2 节点潮流方程（PyTorch 张量） ==============
def power_flow_eq(V1, V2, theta2, Pg1_pu, Qg1, P_load_pu, Q_load_pu, R_pu, X_pu):
    Y = 1 / complex(R_pu, X_pu)
    G_pu, B_line_pu = Y.real, Y.imag

    theta12 = -theta2
    cos12 = torch.cos(theta12)
    sin12 = torch.sin(theta12)
    V1V2 = V1 * V2

    P_inj1 = G_pu * V1**2 - V1V2 * (G_pu * cos12 + B_line_pu * sin12)
    Q_inj1 = -B_line_pu * V1**2 - V1V2 * (G_pu * sin12 - B_line_pu * cos12)
    P_inj2 = G_pu * V2**2 - V1V2 * (G_pu * cos12 - B_line_pu * sin12)
    Q_inj2 = -B_line_pu * V2**2 + V1V2 * (G_pu * sin12 + B_line_pu * cos12)

    P_res1 = P_inj1 - Pg1_pu
    Q_res1 = Q_inj1 - Qg1
    P_res2 = P_inj2 + P_load_pu
    Q_res2 = Q_inj2 + Q_load_pu
    return P_res1, Q_res1, P_res2, Q_res2

# ============== 优先 Appsi-Ipopt 的 Pyomo 求解（含初值钳制） ==============
def solve_with_ipopt_2bus(init, Pg1_pu, P_load_pu, Q_load_pu, R_pu, X_pu,
                          Vmin, Vmax, Qg_min, Qg_max,
                          ipopt_max_iter=300, ipopt_tol=1e-8, silent=True):
    """
    init: dict {'V1','V2','theta2','Qg1'}
    返回: (ok, sol_dict, solver_results)
    """
    try:
        # ====== 初值钳制，避免 W1002 并提高内点法稳健性 ======
        V1_init = float(np.clip(init['V1'], Vmin + 1e-4, Vmax - 1e-4))
        V2_init = float(np.clip(init['V2'], Vmin + 1e-4, Vmax - 1e-4))
        theta2_init = float(np.clip(init['theta2'], -math.pi/2 + 1e-4, math.pi/2 - 1e-4))
        Qg1_init = float(np.clip(init['Qg1'], Qg_min + 1e-4, Qg_max - 1e-4))

        # ====== 建模 ======
        model = pyo.ConcreteModel()
        model.V1 = pyo.Var(bounds=(Vmin, Vmax), initialize=V1_init)
        model.V2 = pyo.Var(bounds=(Vmin, Vmax), initialize=V2_init)
        model.theta2 = pyo.Var(bounds=(-math.pi/2, math.pi/2), initialize=theta2_init)
        model.Qg1 = pyo.Var(bounds=(Qg_min, Qg_max), initialize=Qg1_init)

        # 常量
        Y = 1 / complex(R_pu, X_pu)
        G = float(Y.real)
        B = float(Y.imag)

        def p_res1(m):
            return G*m.V1**2 - m.V1*m.V2*(G*pyo.cos(-m.theta2) + B*pyo.sin(-m.theta2)) - Pg1_pu == 0

        def q_res1(m):
            return -B*m.V1**2 - m.V1*m.V2*(G*pyo.sin(-m.theta2) - B*pyo.cos(-m.theta2)) - m.Qg1 == 0

        def p_res2(m):
            return G*m.V2**2 - m.V1*m.V2*(G*pyo.cos(-m.theta2) - B*pyo.sin(-m.theta2)) + P_load_pu == 0

        def q_res2(m):
            return -B*m.V2**2 + m.V1*m.V2*(G*pyo.sin(-m.theta2) + B*pyo.cos(-m.theta2)) + Q_load_pu == 0

        model.c1 = pyo.Constraint(rule=p_res1)
        model.c2 = pyo.Constraint(rule=q_res1)
        model.c3 = pyo.Constraint(rule=p_res2)
        model.c4 = pyo.Constraint(rule=q_res2)

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
        solver.options['acceptable_tol'] = max(ipopt_tol*10, 1e-6)
        solver.options['max_iter'] = ipopt_max_iter
        solver.options['mu_strategy'] = "adaptive"
        solver.options['linear_solver'] = "mumps"
        solver.options['hessian_approximation'] = "limited-memory"

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
def compute_feasible_region_ipopt():
    conn = init_db()
    S_base = mpc["baseMVA"]
    P_load_pu = mpc["bus"][1][2] / S_base
    Q_load_pu = mpc["bus"][1][3] / S_base
    R_pu = mpc["branch"][0][2]
    X_pu = mpc["branch"][0][3]

    num_points = 100
    Pg1_points_MW = np.linspace(440, 461.5, num_points)

    with open("optimization_log.csv", "w") as log_file:
        log_file.write("Pg1_MW,Start_V1,Start_V2,Best_V1,Best_V2,Final_Loss,Epochs,Compute_Time,IpoptOk\n")

        for Pg1_MW in tqdm(Pg1_points_MW, desc="IPOPT 计算"):
            Pg1_pu = Pg1_MW / S_base
            V1, V2, theta2, Qg1 = initialize_variables(Pg1_pu)
            init = {'V1': V1.item(), 'V2': V2.item(), 'theta2': theta2.item(), 'Qg1': Qg1.item()}

            start_time = time.time()

            ok, sol, res = solve_with_ipopt_2bus(
                init=init, Pg1_pu=Pg1_pu, P_load_pu=P_load_pu, Q_load_pu=Q_load_pu,
                R_pu=R_pu, X_pu=X_pu, Vmin=float('-inf'), Vmax=float('inf'),  # 不限制约束范围
                Qg_min=float('-inf'), Qg_max=float('inf'),
                ipopt_max_iter=500, ipopt_tol=1e-8, silent=True
            )

            end_time = time.time()
            compute_time = end_time - start_time

            # 记录结果
            if ok and sol is not None:
                V1_val, V2_val = sol['V1'], sol['V2']
                best_loss = 0.0  # 可行解损失不需要设置，只记录结果
                log_line = f"{Pg1_MW},{init['V1']:.4f},{init['V2']:.4f},{V1_val:.4f},{V2_val:.4f},{best_loss:.6e},{0},{compute_time:.2f},{1}"
                conn.execute(
                    "INSERT INTO feasible_points (Pg1_MW, V1, V2, loss, epoch_count, compute_time) VALUES (?, ?, ?, ?, ?, ?)",
                    (Pg1_MW, V1_val, V2_val, best_loss, 0, compute_time)
                )
                print(f"[OK] Pg1={Pg1_MW:.1f}MW 收敛: V1={V1_val:.4f}, V2={V2_val:.4f}, time={compute_time:.2f}s")
            else:
                best_loss = float('inf') # 未找到解的情况
                log_line = f"{Pg1_MW},{init['V1']:.4f},{init['V2']:.4f},{0},{0},{best_loss:.6e},{0},{compute_time:.2f},{0}"

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
        c.execute("SELECT * FROM partial_results")
        partial_data = c.fetchall()
        conn.close()

        if not feasible_data and not partial_data:
            print("未找到任何结果！请查看日志文件分析原因")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        if feasible_data:
            Pg1 = [row[0] for row in feasible_data]
            V1 = [row[1] for row in feasible_data]
            V2 = [row[2] for row in feasible_data]
            losses = [row[3] for row in feasible_data]
            scatter = ax1.scatter(V1, V2, c=Pg1, cmap="viridis", s=20, alpha=0.2)
            fig.colorbar(scatter, ax=ax1, label="Pg1 (MW)")
            ax1.set_xlabel("V1 (p.u.)")
            ax1.set_ylabel("V2 (p.u.)")
            ax1.grid(True, linestyle='--', alpha=0.1)
            ax1.set_title("收敛解电压分布")
            ax1.set_xlim(0.94,0.96)
            ax1.set_ylim(0.94, 1.05)

        all_Pg1, all_loss = [], []
        if partial_data:
            partial_Pg1 = [row[0] for row in partial_data]
            partial_loss = [row[1] for row in partial_data]
            all_Pg1.extend(partial_Pg1); all_loss.extend(partial_loss)
            ax2.scatter(partial_Pg1, partial_loss, c='red', s=20, alpha=0.2, label="未收敛")
        if feasible_data:
            all_Pg1.extend(Pg1); all_loss.extend(losses)
            ax2.scatter(Pg1, losses, c='green', s=20, alpha=0.6, label="可行")

        if all_Pg1:
            ax2.set_xlabel("Pg1 (MW)")
            ax2.set_ylabel("最终损失值")
            ax2.set_yscale('log')
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.set_title("不同 Pg1 的收敛损失")
            ax2.legend()

        plt.tight_layout()
        plt.savefig("ipopt_results.png", dpi=300)
        plt.show()
    except sqlite3.OperationalError as e:
        print(f"数据库查询错误: {e}")
        conn.close()
    except Exception as e:
        print(f"可视化错误: {e}")
        if 'conn' in locals() and conn: conn.close()

# ============== 入口 ==============
if __name__ == "__main__":
    if os.path.exists('optimization_log.csv'):
        os.remove('optimization_log.csv')
    compute_feasible_region_ipopt()
    visualize_results()