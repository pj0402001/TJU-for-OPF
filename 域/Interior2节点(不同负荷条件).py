import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib

# ===== 设置中文显示 =====
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===== 系统参数（基于WB2模型）=====
S_base = 100  # MVA
V_base = 230  # kV
Z_base = V_base**2 / S_base

# 线路参数（标幺值）
R_pu = 0.04
X_pu = 0.2
Y = 1 / complex(R_pu, X_pu)
G_pu, B_pu = Y.real, Y.imag

# 发电机参数
P_max_pu = 600 / S_base  # 600 MW -> 6.0 pu
P_min_pu = 0
Q_max_pu = 400 / S_base  # 400 Mvar -> 4.0 pu
Q_min_pu = -400 / S_base  # -400 Mvar -> -4.0 pu

# ===== 负荷参数=====
λ = 1.22  # 负荷缩放系数 1.22时找不到解


P_load_pu = λ * (350 / S_base)  # 350 MW -> 3.5 pu
Q_load_pu = λ * (-350 / S_base)  # -350 Mvar -> -3.5 pu

# 电压约束
Vmin, Vmax = 0.95, 1.05  # 正常电压范围

# ===== 扫描Pg1并计算Qg1 =====
def calculate_pq_points():
    Pg1_points = np.linspace(400, 600, 10000)  # 0-600 MW
    feasible_points = []

    for Pg1_MW in Pg1_points:
        try:
            model = gp.Model("OPF_Pg1_Scan")
            model.setParam("OutputFlag", 0)
            model.setParam("NonConvex", 2)  #MIP求解器
            
            # 变量定义
            V1 = model.addVar(lb=Vmin, ub=Vmax, name="V1")
            V2 = model.addVar(lb=Vmin, ub=Vmax, name="V2")
            theta2 = model.addVar(lb=-math.pi/2, ub=math.pi/2, name="theta2")
            Qg1 = model.addVar(lb=Q_min_pu, ub=Q_max_pu, name="Qg1")
            
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
            
            # 功率平衡方程
            Pg1_pu = Pg1_MW / S_base
            P_inj1 = G_pu * V1**2 - V1V2 * (G_pu * cos12 + B_pu * sin12)
            model.addConstr(P_inj1 == Pg1_pu)
            
            Q_inj1 = -B_pu * V1**2 - V1V2 * (G_pu * sin12 - B_pu * cos12)
            model.addConstr(Qg1 == Q_inj1)
            
            P_inj2 = G_pu * V2**2 - V1V2 * (G_pu * cos12 - B_pu * sin12)
            model.addConstr(P_inj2 == -P_load_pu)
            
            Q_inj2 = -B_pu * V2**2 + V1V2 * (G_pu * sin12 + B_pu * cos12)
            model.addConstr(Q_inj2 == -Q_load_pu)
            
            # 求解
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                Qg1_Mvar = Qg1.X * S_base  # 转换为Mvar
                feasible_points.append((Pg1_MW, Qg1_Mvar))
                
        except Exception as e:
            print(f"Pg1={Pg1_MW:.1f}MW: {str(e)}")
    
    return np.array(feasible_points)

# ===== 计算并绘图 =====
feasible_points = calculate_pq_points()

# 可视化（散点图）
plt.figure(figsize=(10, 6))
if len(feasible_points) > 0:
    plt.scatter(feasible_points[:, 0], feasible_points[:, 1], 
                s=10, c='blue', alpha=0.7, label=f"可行点 (λ={λ})")
    
    plt.xlabel("发电机有功功率 Pg1 (MW)")
    plt.ylabel("发电机无功功率 Qg1 (Mvar)")
    plt.title(f"Pg1-Qg1  (负荷参数 λ={λ})", pad=20)
    plt.legend()
    plt.grid(True)
    plt.xlim(400, 600)
    plt.ylim(0, 350)
    plt.savefig("Pg1_Qg1_scatter.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("未找到可行解！请检查参数或约束条件。")

# 输出当前负荷参数
print(f"\n当前负荷参数：λ={λ}")
print(f"有功负荷 P_load = {λ*350} MW")
print(f"无功负荷 Q_load = {λ*-350} Mvar")