import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
import sqlite3
import pandas as pd
import time
from datetime import datetime
import json
import os
from io import StringIO
import sys

# 完全抑制警告
warnings.filterwarnings('ignore')
import logging
logging.getLogger('pyomo.opt').setLevel(logging.ERROR)

class ACOPFAnalyzer:
    def __init__(self, config_file='ac_opf_config.json'):
        """初始化AC-OPF分析器"""
        self.config = self.load_config(config_file)
        self.setup_system_data()
        self.setup_database()
        
    def load_config(self, config_file):
        """加载配置文件"""
        default_config = {
            "system": {
                "baseMVA": 100,
                "thermal_limit_factor": 1.0,
            },
            "calculation": {
                "n_points":300,  # 减少点数以提高速度
                "batch_size": 100,
                "save_interval": 100
            },
            "solver": {
                "max_iter": 100,
                "tol": 1e-4,
                "acceptable_tol": 1e-3,
                "print_level": 0
            },
            "debug": {
                "enabled": True,
                "verbose": False,
                "save_infeasible": False
            },
            "database": {
                "name": "ac_opf_results.db",
                "auto_clear": True
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self.update_config(default_config, user_config)
        else:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            print(f"📝 已创建配置文件: {config_file}")
            
        return default_config
    
    def update_config(self, default, user):
        """递归更新配置"""
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self.update_config(default[key], value)
                else:
                    default[key] = value
    
    def setup_database(self):
        """设置数据库"""
        self.db_name = self.config['database']['name']
        
        if os.path.exists(self.db_name) and self.config['database'].get('auto_clear', True):
            try:
                temp_conn = sqlite3.connect(self.db_name)
                temp_conn.execute('DROP TABLE IF EXISTS feasible_solutions')
                temp_conn.execute('DROP TABLE IF EXISTS progress') 
                temp_conn.execute('DROP TABLE IF EXISTS analysis_config')
                temp_conn.commit()
                temp_conn.close()
                print(f"🗑️  已清除旧数据库数据")
            except Exception as e:
                print(f"⚠️  清理数据库时出现问题: {e}")
            
        self.conn = sqlite3.connect(self.db_name)
        
        # 创建详细结果表，包含所有发电机和母线信息
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS feasible_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                p1_mw REAL, p2_mw REAL, p3_mw REAL,
                q1_mvar REAL, q2_mvar REAL, q3_mvar REAL,
                v1_pu REAL, v2_pu REAL, v3_pu REAL, v4_pu REAL, v5_pu REAL, 
                v6_pu REAL, v7_pu REAL, v8_pu REAL, v9_pu REAL,
                theta1_deg REAL, theta2_deg REAL, theta3_deg REAL, theta4_deg REAL, theta5_deg REAL,
                theta6_deg REAL, theta7_deg REAL, theta8_deg REAL, theta9_deg REAL,
                total_load REAL, thermal_limit_factor REAL,
                calculation_time REAL, timestamp TEXT
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY,
                total_points INTEGER,
                completed_points INTEGER,
                feasible_count INTEGER,
                start_time TEXT,
                last_update TEXT
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_config (
                id INTEGER PRIMARY KEY,
                thermal_limit_factor REAL,
                p2_min REAL, p2_max REAL,
                p3_min REAL, p3_max REAL,
                n_points INTEGER, total_load REAL,
                analysis_start_time TEXT
            )
        ''')
        
        self.conn.commit()
        print(f"📊 数据库已设置: {self.db_name}")
    
    def setup_system_data(self):
        """设置系统数据"""
        self.baseMVA = self.config['system']['baseMVA']
        thermal_factor = self.config['system']['thermal_limit_factor']
        
        # 母线数据
        self.bus = np.array([
            [1, 3, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [2, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [3, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [4, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [5, 1, 54, 18, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [6, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [7, 1, 60, 21, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [8, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [9, 1, 75, 30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        ])
        
        # 发电机数据
        self.gen = np.array([
            [1, 0, 0, 300, -5, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 163, 0, 300, -5, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 85, 0, 300, -5, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        # 支路数据
        branch_base = np.array([
            [1, 4, 0, 0.0576, 0, 250, 250, 250, 0, 0, 1, -360, 360],
            [4, 5, 0.017, 0.092, 0.158, 250, 250, 250, 0, 0, 1, -360, 360],
            [5, 6, 0.039, 0.17, 0.358, 150, 150, 150, 0, 0, 1, -360, 360],
            [3, 6, 0, 0.0586, 0, 300, 300, 300, 0, 0, 1, -360, 360],
            [6, 7, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
            [7, 8, 0.0085, 0.072, 0.149, 250, 250, 250, 0, 0, 1, -360, 360],
            [8, 2, 0, 0.0625, 0, 250, 250, 250, 0, 0, 1, -360, 360],
            [8, 9, 0.032, 0.161, 0.306, 250, 250, 250, 0, 0, 1, -360, 360],
            [9, 4, 0.01, 0.085, 0.176, 250, 250, 250, 0, 0, 1, -360, 360],
        ])
        
        self.branch = branch_base.copy()
        self.branch[:, 5] = branch_base[:, 5] * thermal_factor
        self.branch[:, 6] = branch_base[:, 6] * thermal_factor
        self.branch[:, 7] = branch_base[:, 7] * thermal_factor
        
        self.setup_system_matrices()
        
        # 设置扫描范围
        self.p2_range = [self.gen_pmin[1] * self.baseMVA, self.gen_pmax[1] * self.baseMVA]
        self.p3_range = [self.gen_pmin[2] * self.baseMVA, self.gen_pmax[2] * self.baseMVA]
        
        total_load = sum(self.bus[:, 2])
        print(f"⚡ 系统配置:")
        print(f"   基准功率: {self.baseMVA} MVA")
        print(f"   总有功负荷: {total_load:.1f} MW")
        print(f"   线路容量缩放因子: {thermal_factor:.2f}")
        print(f"   发电机约束:")
        print(f"     P1∈[{self.gen_pmin[0]*self.baseMVA:.0f},{self.gen_pmax[0]*self.baseMVA:.0f}]MW")
        print(f"     P2∈[{self.p2_range[0]:.0f},{self.p2_range[1]:.0f}]MW")
        print(f"     P3∈[{self.p3_range[0]:.0f},{self.p3_range[1]:.0f}]MW")
    
    def setup_system_matrices(self):
        """建立系统矩阵"""
        self.n_bus = self.bus.shape[0]
        self.n_gen = self.gen.shape[0]
        self.n_branch = self.branch.shape[0]
        
        self.gen_buses = [int(self.gen[i, 0]) - 1 for i in range(self.n_gen)]
        self.Pd = [float(self.bus[i, 2] / self.baseMVA) for i in range(self.n_bus)]
        self.Qd = [float(self.bus[i, 3] / self.baseMVA) for i in range(self.n_bus)]
        self.Vmax = [float(self.bus[i, 11]) for i in range(self.n_bus)]
        self.Vmin = [float(self.bus[i, 12]) for i in range(self.n_bus)]
        
        self.gen_pmax = [float(self.gen[i, 8] / self.baseMVA) for i in range(self.n_gen)]
        self.gen_pmin = [float(self.gen[i, 9] / self.baseMVA) for i in range(self.n_gen)]
        self.gen_qmax = [float(self.gen[i, 3] / self.baseMVA) for i in range(self.n_gen)]
        self.gen_qmin = [float(self.gen[i, 4] / self.baseMVA) for i in range(self.n_gen)]
        
        self.bus_has_gen = [False] * self.n_bus
        self.gen_map = [-1] * self.n_bus
        for i, gb in enumerate(self.gen_buses):
            self.bus_has_gen[gb] = True
            self.gen_map[gb] = i
        
        # 构建导纳矩阵
        Ybus = np.zeros((self.n_bus, self.n_bus), dtype=complex)
        for br in self.branch:
            f = int(br[0]) - 1
            t = int(br[1]) - 1
            r, x, b = br[2:5]
            z = complex(r, x)
            y = 1/z if abs(z) > 1e-8 else 1/complex(0, x)
            b_shunt = complex(0, b/2)
            
            Ybus[f, f] += y + b_shunt
            Ybus[t, t] += y + b_shunt
            Ybus[f, t] -= y
            Ybus[t, f] -= y
        
        self.G = [[float(Ybus[i, j].real) for j in range(self.n_bus)] for i in range(self.n_bus)]
        self.B = [[float(Ybus[i, j].imag) for j in range(self.n_bus)] for i in range(self.n_bus)]
    
    def solve_feasibility(self, p2_fixed, p3_fixed):
        """求解单个点的可行性"""
        start_time = time.time()
        
        try:
            # 重定向输出到字符串，避免控制台输出
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            model = pyo.ConcreteModel()
            
            # 定义集合
            model.buses = pyo.RangeSet(1, self.n_bus)
            model.gens = pyo.RangeSet(1, self.n_gen)
            model.branches = pyo.RangeSet(1, self.n_branch)
            
            # 定义变量
            model.Vm = pyo.Var(model.buses, bounds=lambda m, i: (self.Vmin[i-1], self.Vmax[i-1]), initialize=1.0)
            model.Va = pyo.Var(model.buses, bounds=(-math.pi/4, math.pi/4), initialize=0.0)
            
            model.Pg = pyo.Var(model.gens, bounds=lambda m, g: (self.gen_pmin[g-1], self.gen_pmax[g-1]))
            model.Qg = pyo.Var(model.gens, bounds=lambda m, g: (self.gen_qmin[g-1], self.gen_qmax[g-1]), initialize=0.0)
            
            # 目标函数（最小化成本，这里设为0）
            model.cost = pyo.Objective(expr=0, sense=pyo.minimize)
            
            # 固定P2和P3
            model.pg2_fixed = pyo.Constraint(expr=model.Pg[2] == p2_fixed)
            model.pg3_fixed = pyo.Constraint(expr=model.Pg[3] == p3_fixed)
            
            # 功率平衡约束
            def power_balance_p(m, i):
                Pi = sum(m.Vm[i] * m.Vm[j] * (self.G[i-1][j-1] * pyo.cos(m.Va[i] - m.Va[j]) + 
                                               self.B[i-1][j-1] * pyo.sin(m.Va[i] - m.Va[j])) 
                         for j in m.buses)
                Pg_inject = m.Pg[self.gen_map[i-1] + 1] if self.bus_has_gen[i-1] else 0
                return Pg_inject - self.Pd[i-1] == Pi
            
            def power_balance_q(m, i):
                Qi = sum(m.Vm[i] * m.Vm[j] * (self.G[i-1][j-1] * pyo.sin(m.Va[i] - m.Va[j]) - 
                                               self.B[i-1][j-1] * pyo.cos(m.Va[i] - m.Va[j])) 
                         for j in m.buses)
                Qg_inject = m.Qg[self.gen_map[i-1] + 1] if self.bus_has_gen[i-1] else 0
                return Qg_inject - self.Qd[i-1] == Qi
            
            model.power_balance_p = pyo.Constraint(model.buses, rule=power_balance_p)
            model.power_balance_q = pyo.Constraint(model.buses, rule=power_balance_q)
            
            # 参考节点约束
            model.ref_bus = pyo.Constraint(expr=model.Va[1] == 0)
            
            # 线路潮流约束
            def branch_flow_rule(m, br_idx):
                br = self.branch[br_idx-1]
                f_bus = int(br[0])
                t_bus = int(br[1])
                r, x, b = br[2:5]
                
                # 计算线路潮流
                Vf = m.Vm[f_bus]
                Vt = m.Vm[t_bus]
                theta_f = m.Va[f_bus]
                theta_t = m.Va[t_bus]
                theta_ft = theta_f - theta_t
                
                # 线路导纳
                y = 1/complex(r, x) if abs(complex(r, x)) > 1e-8 else complex(0, 1/x)
                g = y.real
                b_y = y.imag
                
                # 有功潮流
                Pf = g * Vf**2 - Vf * Vt * (g * pyo.cos(theta_ft) + b_y * pyo.sin(theta_ft))
                
                # 热稳定约束
                thermal_limit = br[5] / self.baseMVA
                return Pf**2 <= thermal_limit**2
            
            model.thermal_limits = pyo.Constraint(model.branches, rule=branch_flow_rule)
            
            # 求解器设置
            solver = SolverFactory('ipopt')
            solver.options.update(self.config['solver'])
            
            # 求解
            results = solver.solve(model, tee=False)
            
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            calc_time = time.time() - start_time
            
            if str(results.solver.termination_condition) == 'optimal':
                # 提取所有变量值
                solution_data = {}
                
                # 发电机有功无功
                for i in range(1, 4):
                    solution_data[f'p{i}_mw'] = pyo.value(model.Pg[i]) * self.baseMVA
                    solution_data[f'q{i}_mvar'] = pyo.value(model.Qg[i]) * self.baseMVA
                
                # 电压幅值和相角
                for i in range(1, 10):
                    solution_data[f'v{i}_pu'] = pyo.value(model.Vm[i])
                    solution_data[f'theta{i}_deg'] = pyo.value(model.Va[i]) * 180 / math.pi
                
                return True, solution_data, calc_time
            else:
                return False, None, calc_time
                
        except Exception as e:
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return False, None, time.time() - start_time
    
    def save_result(self, p2_mw, p3_mw, solution_data, calc_time):
        """保存结果到数据库"""
        total_load = sum(self.bus[:, 2])
        thermal_factor = self.config['system']['thermal_limit_factor']
        timestamp = datetime.now().isoformat()
        
        # 准备数据
        data = [
            solution_data.get('p1_mw', 0), p2_mw, p3_mw,
            solution_data.get('q1_mvar', 0), solution_data.get('q2_mvar', 0), solution_data.get('q3_mvar', 0)
        ]
        
        # 添加电压
        for i in range(1, 10):
            data.append(solution_data.get(f'v{i}_pu', 1.0))
        
        # 添加相角
        for i in range(1, 10):
            data.append(solution_data.get(f'theta{i}_deg', 0.0))
        
        # 添加其他信息
        data.extend([total_load, thermal_factor, calc_time, timestamp])
        
        # 插入数据库
        placeholders = ','.join(['?'] * len(data))
        self.conn.execute(f'''
            INSERT INTO feasible_solutions 
            (p1_mw, p2_mw, p3_mw, q1_mvar, q2_mvar, q3_mvar,
             v1_pu, v2_pu, v3_pu, v4_pu, v5_pu, v6_pu, v7_pu, v8_pu, v9_pu,
             theta1_deg, theta2_deg, theta3_deg, theta4_deg, theta5_deg,
             theta6_deg, theta7_deg, theta8_deg, theta9_deg,
             total_load, thermal_limit_factor, calculation_time, timestamp)
            VALUES ({placeholders})
        ''', data)
        
        self.conn.commit()
    
    def save_analysis_config(self, n_points, start_time):
        """保存分析配置"""
        total_load = sum(self.bus[:, 2])
        thermal_factor = self.config['system']['thermal_limit_factor']
        
        self.conn.execute('''
            INSERT OR REPLACE INTO analysis_config 
            (id, thermal_limit_factor, p2_min, p2_max, p3_min, p3_max, n_points, total_load, analysis_start_time)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (thermal_factor, self.p2_range[0], self.p2_range[1], 
              self.p3_range[0], self.p3_range[1], n_points, total_load, start_time))
        self.conn.commit()
    
    def update_progress(self, total_points, completed_points, feasible_count, start_time):
        """更新进度"""
        self.conn.execute('''
            INSERT OR REPLACE INTO progress 
            (id, total_points, completed_points, feasible_count, start_time, last_update)
            VALUES (1, ?, ?, ?, ?, ?)
        ''', (total_points, completed_points, feasible_count, start_time, datetime.now().isoformat()))
        self.conn.commit()
    
    def run_analysis(self):
        """运行完整分析"""
        print(f"\n🚀 开始AC-OPF可行域分析")
        print("="*60)
        
        n_points = self.config['calculation']['n_points']
        batch_size = self.config['calculation']['batch_size']
        save_interval = self.config['calculation']['save_interval']
        
        # 生成扫描点
        p2_values = np.linspace(self.p2_range[0], self.p2_range[1], n_points)
        p3_values = np.linspace(self.p3_range[0], self.p3_range[1], n_points)
        
        total_points = len(p2_values) * len(p3_values)
        
        print(f"📊 计算规模:")
        print(f"   网格: {n_points}×{n_points} = {total_points:,} 点")
        print(f"   P2扫描范围: {self.p2_range[0]:.1f} - {self.p2_range[1]:.1f} MW")
        print(f"   P3扫描范围: {self.p3_range[0]:.1f} - {self.p3_range[1]:.1f} MW")
        print(f"   线路容量缩放因子: {self.config['system']['thermal_limit_factor']:.2f}")
        
        start_time = datetime.now().isoformat()
        self.save_analysis_config(n_points, start_time)
        
        current_point = 0
        feasible_count = 0
        
        print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔄 计算进行中...")
        
        for i, p2 in enumerate(p2_values):
            for j, p3 in enumerate(p3_values):
                current_point += 1
                
                # 转换为标幺值并求解
                p2_pu = p2 / self.baseMVA
                p3_pu = p3 / self.baseMVA
                
                is_feasible, solution_data, calc_time = self.solve_feasibility(p2_pu, p3_pu)
                
                if is_feasible and solution_data:
                    self.save_result(p2, p3, solution_data, calc_time)
                    feasible_count += 1
                
                # 更新进度
                if current_point % save_interval == 0 or current_point == total_points:
                    self.update_progress(total_points, current_point, feasible_count, start_time)
                    
                    progress = 100 * current_point / total_points
                    success_rate = 100 * feasible_count / current_point if current_point > 0 else 0
                    est_time_left = self.estimate_time_left(current_point, total_points, start_time)
                    
                    print(f"   进度: {current_point:,}/{total_points:,} ({progress:.1f}%) | "
                          f"可行解: {feasible_count:,} ({success_rate:.1f}%) | "
                          f"剩余: {est_time_left}")
        
        print(f"\n✅ 分析完成!")
        print(f"   总点数: {total_points:,}")
        print(f"   可行解: {feasible_count:,}")
        print(f"   成功率: {100*feasible_count/total_points:.1f}%")
        print(f"   结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def estimate_time_left(self, completed, total, start_time):
        """估算剩余时间"""
        if completed == 0:
            return "计算中..."
        
        elapsed = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
        rate = completed / elapsed
        remaining_points = total - completed
        remaining_seconds = remaining_points / rate
        
        if remaining_seconds < 60:
            return f"{remaining_seconds:.0f}秒"
        elif remaining_seconds < 3600:
            return f"{remaining_seconds/60:.0f}分钟"
        else:
            return f"{remaining_seconds/3600:.1f}小时"
    
    def export_to_csv(self, filename="ac_opf_results.csv"):
        """导出结果为CSV文件"""
        try:
            # 从数据库读取所有数据
            df = pd.read_sql_query("SELECT * FROM feasible_solutions", self.conn)
            
            if len(df) == 0:
                print("⚠️ 没有可行解数据可导出")
                return False
            
            # 保存为CSV
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"💾 结果已导出到: {filename}")
            print(f"   包含 {len(df)} 个可行解")
            return True
            
        except Exception as e:
            print(f"❌ 导出CSV时出错: {e}")
            return False
    
    def plot_results(self, limit=None):
        """绘制结果"""
        query = "SELECT p1_mw, p2_mw, p3_mw FROM feasible_solutions"
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, self.conn)
        
        if len(df) == 0:
            print("⚠️ 没有可行解数据用于绘图")
            return
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['p1_mw'], df['p2_mw'], c=df['p3_mw'], 
                             cmap='viridis', s=6, alpha=0.7, edgecolors='none')
        
        plt.colorbar(scatter, label='发电机3功率 P3 (MW)')
        plt.xlabel('平衡机组功率 P1 (MW)', fontsize=14)
        plt.ylabel('发电机2功率 P2 (MW)', fontsize=14)
        
        # 获取配置信息
        cursor = self.conn.execute('SELECT * FROM analysis_config WHERE id = 1')
        config_data = cursor.fetchone()
        
        if config_data:
            thermal_factor = config_data[1]
            total_load = config_data[7]
            title = f'线路容量因子{thermal_factor:.2f}下的AC-OPF可行域\n(总负荷{total_load:.1f}MW，共{len(df):,}个可行解)'
        else:
            title = f'AC-OPF可行域 (共{len(df):,}个可行解)'
            
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\n📈 绘图数据统计:")
        print(f"   可行解数量: {len(df):,}")
        print(f"   P1范围: {df['p1_mw'].min():.2f} - {df['p1_mw'].max():.2f} MW")
        print(f"   P2范围: {df['p2_mw'].min():.2f} - {df['p2_mw'].max():.2f} MW")
        print(f"   P3范围: {df['p3_mw'].min():.2f} - {df['p3_mw'].max():.2f} MW")
    
    def get_statistics(self):
        """获取统计信息"""
        cursor = self.conn.execute('SELECT COUNT(*) FROM feasible_solutions')
        feasible_count = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT * FROM progress WHERE id = 1')
        progress = cursor.fetchone()
        
        cursor = self.conn.execute('SELECT * FROM analysis_config WHERE id = 1')
        config_data = cursor.fetchone()
        
        stats = {
            'feasible_solutions': feasible_count,
            'total_points': progress[1] if progress else 0,
            'completed_points': progress[2] if progress else 0,
            'success_rate': 100 * feasible_count / progress[2] if progress and progress[2] > 0 else 0,
        }
        
        if config_data:
            stats.update({
                'thermal_limit_factor': config_data[1],
                'total_load': config_data[7]
            })
        
        return stats
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()

# 使用示例
if __name__ == "__main__":
    print("🔧 AC-OPF可行域分析工具")
    print("="*55)
    
    analyzer = ACOPFAnalyzer()
    
    try:
        # 显示当前配置
        print(f"📋 当前配置:")
        print(f"   网格大小: {analyzer.config['calculation']['n_points']}×{analyzer.config['calculation']['n_points']}")
        print(f"   线路容量缩放因子: {analyzer.config['system']['thermal_limit_factor']}")
        print(f"   扫描范围: P2∈[{analyzer.p2_range[0]:.0f},{analyzer.p2_range[1]:.0f}], P3∈[{analyzer.p3_range[0]:.0f},{analyzer.p3_range[1]:.0f}]")  
        
        # 菜单选择
        while True:
            print(f"\n📞 请选择操作:")
            print("1. 运行可行域分析")
            print("2. 绘制结果图")
            print("3. 查看统计信息")
            print("4. 导出为CSV文件")
            print("5. 退出")
            
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == '1':
                analyzer.run_analysis()
            elif choice == '2':
                limit = input("显示数据点数限制 (回车显示全部): ").strip()
                limit = int(limit) if limit.isdigit() else None
                analyzer.plot_results(limit)
            elif choice == '3':
                stats = analyzer.get_statistics()
                print(f"\n📊 统计信息:")
                print(f"   可行解: {stats['feasible_solutions']:,}")
                print(f"   总计算点: {stats.get('completed_points', 0):,}")
                print(f"   成功率: {stats.get('success_rate', 0):.1f}%")
                if 'thermal_limit_factor' in stats:
                    print(f"   线路容量因子: {stats['thermal_limit_factor']:.2f}")
                    print(f"   总负荷: {stats['total_load']:.1f} MW")
            elif choice == '4':
                filename = input("输入CSV文件名 (回车使用默认名): ").strip()
                if not filename:
                    filename = "ac_opf_results.csv"
                analyzer.export_to_csv(filename)
            elif choice == '5':
                break
            else:
                print("❌ 无效选择")
    
    finally:
        analyzer.close()
        print("👋 程序结束")