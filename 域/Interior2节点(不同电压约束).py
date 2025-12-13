import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import math
import sqlite3
from tqdm import tqdm

# ===== 参数设置 =====
mpc = {
    "baseMVA": 100,
    "bus": [
        [1, 3, 0,   0,   0, 0, 1, 0.964, 0,   0, 1, 1.05, 0.95],  # 发电机节点
        [2, 1, 350, -350, 0, 0, 1, 1.0,   -65, 0, 1, 1.05, 0.95]   # 负荷节点
    ],
    "gen": [
        [1, 400, 100, 400, -400, 0.964, 100, 1, 600, 0]  # 发电机数据
    ],
    "branch": [
        [1, 2, 0.04, 0.2, 0, 990000, 0, 0, 0, 0, 1, -360, 360]  # 支路数据
    ]
}

# ===== 数据库初始化 =====
def init_db():
    conn = sqlite3.connect('opf_results.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS feasible_points
                    (Pg1_MW REAL, V1 REAL, V2 REAL)''')
    conn.commit()
    return conn

# ===== 主计算流程 =====
def compute_feasible_region():
    conn = init_db()
    S_base = mpc["baseMVA"]
    
    # 转换为标幺值
    P_load_pu = mpc["bus"][1][2] / S_base
    Q_load_pu = mpc["bus"][1][3] / S_base
    R_pu = mpc["branch"][0][2]
    X_pu = mpc["branch"][0][3]
    Y = 1 / complex(R_pu, X_pu)
    G_pu, B_line_pu = Y.real, Y.imag
    
    # 电压范围
    Vmin, Vmax = 0.95, 1.05
    
    # 扫描 Pg1（0-600 MW）
    num_points =500
    Pg1_points_MW = np.linspace(440, 460, num_points)
    
    for Pg1_MW in tqdm(Pg1_points_MW, desc="计算可行解"):
        Pg1_pu = Pg1_MW / S_base
        
        try:
            model = gp.Model("OPF_WB2")
            model.setParam("OutputFlag", 0)
            model.setParam("NonConvex", 2)
            
            # 变量定义
            V1 = model.addVar(lb=Vmin, ub=Vmax, name="V1")
            V2 = model.addVar(lb=0.95, ub=1.05, name="V2")
            theta2 = model.addVar(lb=-math.pi, ub=math.pi, name="theta2")
            Qg1 = model.addVar(lb=-4.0, ub=4.0, name="Qg1")  # Qg1范围(-400,400)Mvar
            
            # 中间变量
            theta12 = model.addVar(name="theta12")
            cos12 = model.addVar(lb=-1, ub=1, name="cos12")
            sin12 = model.addVar(lb=-1, ub=1, name="sin12")
            V1V2 = model.addVar(name="V1V2")
            
            # 约束条件
            model.addConstr(theta12 == -theta2)
            model.addGenConstrCos(theta12, cos12)
            model.addGenConstrSin(theta12, sin12)
            model.addConstr(V1V2 == V1 * V2)
            
            # 功率平衡
            P_inj1 = G_pu * V1**2 - V1V2 * (G_pu * cos12 + B_line_pu * sin12)
            model.addConstr(P_inj1 == Pg1_pu, "P_balance_1")
            
            Q_inj1 = -B_line_pu * V1**2 - V1V2 * (G_pu * sin12 - B_line_pu * cos12)
            model.addConstr(Q_inj1 == Qg1, "Q_balance_1")
            
            P_inj2 = G_pu * V2**2 - V1V2 * (G_pu * cos12 - B_line_pu * sin12)
            model.addConstr(P_inj2 == -P_load_pu, "P_balance_2")
            
            Q_inj2 = -B_line_pu * V2**2 + V1V2 * (G_pu * sin12 + B_line_pu * cos12)
            model.addConstr(Q_inj2 == -Q_load_pu, "Q_balance_2")
            
            # 求解
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                conn.execute(
                    "INSERT INTO feasible_points VALUES (?,?,?)",
                    (Pg1_MW, V1.X, V2.X)
                )
                if len(conn.execute("SELECT * FROM feasible_points").fetchall()) % 10000 == 0:
                    conn.commit()  # 每10000次提交一次事务
            
        except Exception as e:
            print(f"Error at Pg1={Pg1_MW:.1f} MW: {str(e)}")
            continue
    
    conn.commit()
    conn.close()

# ===== 可视化全部可行解 =====
def visualize_all_feasible_points():
    conn = sqlite3.connect('opf_results.db')
    c = conn.cursor()
    
    # 获取全部可行解
    c.execute("SELECT Pg1_MW, V1, V2 FROM feasible_points")
    data = c.fetchall()
    conn.close()
    
    if not data:
        print("未找到可行解！请检查系统参数或约束条件。")
        return
    
    Pg1, V1, V2 = zip(*data)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(V1, V2, c=Pg1, cmap="viridis", s=5, alpha=0.5)
    plt.colorbar(label="发电机有功功率 Pg1 (MW)")
    plt.xlabel("节点1电压 V1 (p.u.)", fontsize=12)
    plt.ylabel("节点2电压 V2 (p.u.)", fontsize=12)
    plt.xlim(0.94, 0.96)
    plt.ylim(0.94, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"两节点系统可行域 (共 {len(data)} 个可行解)", fontsize=14)
    
    # 保存高清图像
    plt.savefig("feasible_region_full.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    compute_feasible_region()
    visualize_all_feasible_points()