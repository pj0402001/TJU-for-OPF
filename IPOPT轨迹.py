import numpy as np
import torch
import matplotlib.pyplot as plt
import pandapower as pp
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# 创建空的电网
net = pp.create_empty_network()

# 创建母线（bus）
bus1 = pp.create_bus(net, vn_kv=10, name="Bus 1")
bus2 = pp.create_bus(net, vn_kv=10, name="Bus 2")

# 创建发电机（gen）
pp.create_gen(net, bus1, p_mw=0.4417, vm_pu=1.0)

# 创建负载（load）
pp.create_load(net, bus2, p_mw=0.35, q_mvar=0.2)

# 创建输电线（branch）
line = pp.create_line_from_parameters(net, bus1, bus2,
    length_km=1.0,
    r_ohm_per_km=0.045,
    x_ohm_per_km=0.3,
    c_nf_per_km=0.0,    # 电容每公里
    max_i_ka=1.0)  # 最大电流 kA

# 创建电压源（调节电压）
pp.create_ext_grid(net, bus1, vm_pu=1.0)

# 使用 IPOPT 进行求解
def solve_with_ipopt(Pg1_pu, P_load_pu, Q_load_pu, R_pu, X_pu, V1_init, V2_init):
    model = pyo.ConcreteModel()
    
    # 定义变量
    model.V1 = pyo.Var(bounds=(0.95, 1.05), initialize=V1_init)
    model.V2 = pyo.Var(bounds=(0.95, 1.05), initialize=V2_init)
    model.theta2 = pyo.Var(bounds=(-np.pi/2, np.pi/2), initialize=0)
    model.Qg1 = pyo.Var(bounds=(-4.0, 4.0), initialize=0)

    # 定义电流容抗
    Y = 1 / complex(R_pu, X_pu)
    G = float(Y.real)
    B = float(Y.imag)

    # 定义潮流方程
    def p_res1(m):
        return G * m.V1**2 - m.V1 * m.V2 * (G * pyo.cos(-m.theta2) + B * pyo.sin(-m.theta2)) - Pg1_pu == 0

    def q_res1(m):
        return -B * m.V1**2 - m.V1 * m.V2 * (G * pyo.sin(-m.theta2) - B * pyo.cos(-m.theta2)) - m.Qg1 == 0

    def p_res2(m):
        return G * m.V2**2 - m.V1 * m.V2 * (G * pyo.cos(-m.theta2) - B * pyo.sin(-m.theta2)) + P_load_pu == 0

    def q_res2(m):
        return -B * m.V2**2 + m.V1 * m.V2 * (G * pyo.sin(-m.theta2) + B * pyo.cos(-m.theta2)) + Q_load_pu == 0

    # 定义约束
    model.constraint1 = pyo.Constraint(rule=p_res1)
    model.constraint2 = pyo.Constraint(rule=q_res1)
    model.constraint3 = pyo.Constraint(rule=p_res2)
    model.constraint4 = pyo.Constraint(rule=q_res2)

    # 定义目标函数（无需优化，可以设为 0）
    model.obj = pyo.Objective(expr=0.0, sense=pyo.minimize)

    # 求解
    solver = SolverFactory('ipopt')
    solver.options['print_level'] = 5  # 提高打印级别以查看详细信息
    results = solver.solve(model)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        return True, pyo.value(model.V1), pyo.value(model.V2)
    else:
        print("求解失败，终止条件:", results.solver.termination_condition)
        print("详细信息:", results)
        return False, None, None

# 运行 IPOPT 并获取迭代过程
initial_V1, initial_V2 = 0.964, 1.0  # 初始条件
Pg1_pu = 0.4417  # 发电机功率
P_load_pu = 0.35  # 负载功率
Q_load_pu = 0.2  # 负载功率
R_pu = 0.045  # 线路电阻
X_pu = 0.3  # 线路反应

# 进行 IPOPT 求解
converged, final_V1, final_V2 = solve_with_ipopt(Pg1_pu, P_load_pu, Q_load_pu, R_pu, X_pu, initial_V1, initial_V2)

# 打印结果
if converged:
    print(f"求解成功，最终 V1 = {final_V1:.4f}, V2 = {final_V2:.4f}")
else:
    print("求解失败")

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(initial_V1, initial_V2, color='green', s=100, label='Initial Point')  # 初始点
plt.scatter(final_V1, final_V2, color='red', s=100, label='Final Point')  # 收敛点

plt.title('IPOPT Method Results')
plt.xlabel('V1 (pu)')
plt.ylabel('V2 (pu)')
plt.grid()
plt.legend()
plt.xlim(0.94, 1.06)
plt.ylim(0.94, 1.06)
plt.show()