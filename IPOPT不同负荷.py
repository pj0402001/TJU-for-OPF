
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
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# 完全抑制警告
warnings.filterwarnings('ignore')
import logging
logging.getLogger('pyomo.opt').setLevel(logging.ERROR)

class ACOPFAnalyzer:
    def __init__(self, config_file='ac_opf_config.json'):
        """初始化AC-OPF分析器"""
        self.config = self.load_config(config_file)
        self.setup_system_data()  # 先设置系统数据，获取约束范围
        self.setup_database()
        
    def load_config(self, config_file):
        """加载配置文件"""
        default_config = {
            "system": {
                "baseMVA": 100,
                "load_scale_factor": 1.00,  # 负荷缩放因子
                "p_load": [0, 0, 0, 0, 54, 0, 60, 0, 75],  # 各母线有功负荷MW
                "q_load": [0, 0, 0, 0, 18, 0, 21, 0, 30]   # 各母线无功负荷MVar
            },
            "calculation": {
                "n_points": 50,  # 每个维度的点数 (调试用小值)
                "batch_size": 1000,     # 批处理大小
                "save_interval": 100    # 保存间隔
            },
            "solver": {
                "max_iter": 100,
                "tol": 1e-4,
                "acceptable_tol": 1e-3
            },
            "debug": {
                "enabled": True,
                "verbose": True,
                "save_infeasible": False
            },
            "database": {
                "name": "ac_opf_results.db",
                "auto_clear": True  # 自动清理数据库
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 递归更新配置
                self.update_config(default_config, user_config)
        else:
            # 创建默认配置文件
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
        
        # 数据库存在时删除（每次分析都重新开始）
        if os.path.exists(self.db_name):
            os.remove(self.db_name) 
            print(f"🗑️  已清除旧数据库数据")
            
        self.conn = sqlite3.connect(self.db_name)
        
        # 创建结果表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS feasible_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                p2_mw REAL,
                p3_mw REAL,
                p1_mw REAL,
                total_load REAL,
                load_factor REAL,
                calculation_time REAL,
                timestamp TEXT
            )
        ''')
        
        # 创建进度表
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
        
        # 创建配置记录表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_config (
                id INTEGER PRIMARY KEY,
                load_factor REAL,
                p2_min REAL,
                p2_max REAL,
                p3_min REAL,
                p3_max REAL,
                n_points INTEGER,
                total_load REAL,
                analysis_start_time TEXT
            )
        ''')
        
        self.conn.commit()
        print(f"📊 数据库已重置: {self.db_name}")
    
    def setup_system_data(self):
        """设置系统数据"""
        self.baseMVA = self.config['system']['baseMVA']
        load_factor = self.config['system']['load_scale_factor']
        
        # 母线数据
        self.bus = np.array([
            [1, 3, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [2, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [3, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [4, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [5, 1, self.config['system']['p_load'][4]*load_factor, 
                   self.config['system']['q_load'][4]*load_factor, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [6, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [7, 1, self.config['system']['p_load'][6]*load_factor, 
                   self.config['system']['q_load'][6]*load_factor, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [8, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [9, 1, self.config['system']['p_load'][8]*load_factor, 
                   self.config['system']['q_load'][8]*load_factor, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        ])
        
        # 发电机数据
        self.gen = np.array([
            [1, 0, 0, 300, -5, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 163, 0, 300, -5, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 85, 0, 300, -5, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        # 支路数据
        self.branch = np.array([
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
        
        self.setup_system_matrices()
        
        # 自动设置P2, P3扫描范围为约束范围
        self.p2_range = [self.gen_pmin[1] * self.baseMVA, self.gen_pmax[1] * self.baseMVA]
        self.p3_range = [self.gen_pmin[2] * self.baseMVA, self.gen_pmax[2] * self.baseMVA]
        
        # 输出系统信息
        total_load = sum(self.bus[:, 2])
        print(f"⚡ 系统配置:")
        print(f"   基准功率: {self.baseMVA} MVA")
        print(f"   负荷缩放因子: {load_factor:.2f}")
        print(f"   总有功负荷: {total_load:.1f} MW")
        print(f"   发电机约束:")
        print(f"     P1∈[{self.gen_pmin[0]*self.baseMVA:.0f},{self.gen_pmax[0]*self.baseMVA:.0f}]MW")
        print(f"     P2∈[{self.p2_range[0]:.0f},{self.p2_range[1]:.0f}]MW (扫描范围)")
        print(f"     P3∈[{self.p3_range[0]:.0f},{self.p3_range[1]:.0f}]MW (扫描范围)")
    
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
            f_stdout = StringIO()
            f_stderr = StringIO()
            
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                model = pyo.ConcreteModel()
                
                model.buses = pyo.RangeSet(1, self.n_bus)
                model.gens = pyo.RangeSet(1, self.n_gen)
                
                model.Vm = pyo.Var(model.buses, bounds=lambda m, i: (self.Vmin[i-1], self.Vmax[i-1]), 
                                  initialize=1.0)
                model.Va = pyo.Var(model.buses, bounds=(-math.pi/4, math.pi/4), initialize=0.0)
                
                model.Pg = pyo.Var(model.gens, 
                                  bounds=lambda m, g: (self.gen_pmin[g-1], self.gen_pmax[g-1]))
                model.Qg = pyo.Var(model.gens,
                                  bounds=lambda m, g: (self.gen_qmin[g-1], self.gen_qmax[g-1]),
                                  initialize=0.0)
                
                # 智能初始化
                total_load = sum(self.Pd)
                remaining_load = total_load - p2_fixed - p3_fixed
                model.Pg[1].set_value(max(self.gen_pmin[0], min(self.gen_pmax[0], remaining_load)))
                model.Pg[2].set_value(p2_fixed)
                model.Pg[3].set_value(p3_fixed)
                
                model.cost = pyo.Objective(expr=0, sense=pyo.minimize)
                
                model.pg2_fixed = pyo.Constraint(expr=model.Pg[2] == p2_fixed)
                model.pg3_fixed = pyo.Constraint(expr=model.Pg[3] == p3_fixed)
                
                def power_balance_p(m, i):
                    Pi = sum(m.Vm[i] * m.Vm[j] * (self.G[i-1][j-1] * pyo.cos(m.Va[i] - m.Va[j]) + 
                                                   self.B[i-1][j-1] * pyo.sin(m.Va[i] - m.Va[j])) 
                             for j in m.buses)
                    Pg_inject = m.Pg[self.gen_map[i-1] + 1] if self.bus_has_gen[i-1] else 0
                    return Pg_inject - self.Pd[i-1] == Pi
                
                model.power_balance_p = pyo.Constraint(model.buses, rule=power_balance_p)
                
                def power_balance_q(m, i):
                    Qi = sum(m.Vm[i] * m.Vm[j] * (self.G[i-1][j-1] * pyo.sin(m.Va[i] - m.Va[j]) - 
                                                   self.B[i-1][j-1] * pyo.cos(m.Va[i] - m.Va[j])) 
                             for j in m.buses)
                    Qg_inject = m.Qg[self.gen_map[i-1] + 1] if self.bus_has_gen[i-1] else 0
                    return Qg_inject - self.Qd[i-1] == Qi
                
                model.power_balance_q = pyo.Constraint(model.buses, rule=power_balance_q)
                model.ref_bus = pyo.Constraint(expr=model.Va[1] == 0)
                
                # 求解器设置
                # 优先使用 IDAES 的预编译 ipopt（如果已安装 get-extensions）
                solver = None
                try:
                    from idaes import bin_directory
                    ipopt_exe = os.path.join(bin_directory, "ipopt.exe")
                    if os.path.exists(ipopt_exe):
                        solver = SolverFactory("ipopt", executable=ipopt_exe)
                        print(f"[DEBUG] Using IDAES IPOPT at: {ipopt_exe}")
                except Exception:
                    pass

                if solver is None:
                    solver = SolverFactory('ipopt')
                    print(f"[DEBUG] Using system IPOPT at: {solver.executable()}")

                solver.options.clear()
                solver.options.update(self.config['solver'])
                solver.options['print_level'] = 0
                # 避免依赖 MKL 的线性求解器，优先尝试 MUMPS（若该二进制包含）
                solver.options.setdefault('linear_solver', 'mumps')
                
                results = solver.solve(model, tee=False)
                
                calc_time = time.time() - start_time
                
                if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                    try:
                        p1_value = pyo.value(model.Pg[1]) * self.baseMVA
                        if self.gen_pmin[0]*self.baseMVA <= p1_value <= self.gen_pmax[0]*self.baseMVA:
                            return True, p1_value, calc_time
                    except:
                        pass
                
                return False, None, calc_time
                
        except Exception:
            return False, None, time.time() - start_time
    
    def save_result(self, p2_mw, p3_mw, p1_mw, calc_time):
        """保存结果到数据库"""
        total_load = sum(self.bus[:, 2])
        load_factor = self.config['system']['load_scale_factor']
        timestamp = datetime.now().isoformat()
        
        self.conn.execute('''
            INSERT INTO feasible_solutions 
            (p2_mw, p3_mw, p1_mw, total_load, load_factor, calculation_time, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (p2_mw, p3_mw, p1_mw, total_load, load_factor, calc_time, timestamp))
    
    def save_analysis_config(self, n_points, start_time):
        """保存分析配置"""
        total_load = sum(self.bus[:, 2])
        load_factor = self.config['system']['load_scale_factor']
        
        self.conn.execute('''
            INSERT OR REPLACE INTO analysis_config 
            (id, load_factor, p2_min, p2_max, p3_min, p3_max, n_points, total_load, analysis_start_time)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (load_factor, self.p2_range[0], self.p2_range[1], 
              self.p3_range[0], self.p3_range[1], n_points, total_load, start_time))
    
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
        # 确认开始新分析（数据库已在初始化时清理）
        print(f"\n🚀 开始新的AC-OPF可行域分析")
        print("="*60)
        
        n_points = self.config['calculation']['n_points']
        batch_size = self.config['calculation']['batch_size']
        save_interval = self.config['calculation']['save_interval']
        
        # 使用约束范围作为扫描范围
        p2_values = np.linspace(self.p2_range[0], self.p2_range[1], n_points)
        p3_values = np.linspace(self.p3_range[0], self.p3_range[1], n_points)
        
        total_points = len(p2_values) * len(p3_values)
        
        print(f"📊 计算规模:")
        print(f"   网格: {n_points}×{n_points} = {total_points:,} 点")
        print(f"   P2扫描范围: {self.p2_range[0]:.1f} - {self.p2_range[1]:.1f} MW")
        print(f"   P3扫描范围: {self.p3_range[0]:.1f} - {self.p3_range[1]:.1f} MW")
        print(f"   负荷缩放因子: {self.config['system']['load_scale_factor']:.2f}")
        print(f"   批处理: {batch_size} 点/批")
        
        start_time = datetime.now().isoformat()
        self.save_analysis_config(n_points, start_time)
        
        current_point = 0
        feasible_count = 0
        batch_results = []
        
        print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔄 计算进行中...")
        
        for i, p2 in enumerate(p2_values):
            for j, p3 in enumerate(p3_values):
                current_point += 1
                
                # 求解
                p2_pu = p2 / self.baseMVA
                p3_pu = p3 / self.baseMVA
                
                is_feasible, p1_value, calc_time = self.solve_feasibility(p2_pu, p3_pu)
                
                if is_feasible:
                    batch_results.append((p2, p3, p1_value, calc_time))
                    feasible_count += 1
                
                # 批量保存
                if len(batch_results) >= batch_size:
                    for result in batch_results:
                        self.save_result(*result)
                    batch_results = []
                
                # 定期更新进度
                if current_point % save_interval == 0:
                    self.update_progress(total_points, current_point, feasible_count, start_time)
                    
                    # 显示进度
                    progress = 100 * current_point / total_points
                    success_rate = 100 * feasible_count / current_point if current_point > 0 else 0
                    est_time_left = self.estimate_time_left(current_point, total_points, start_time)
                    
                    print(f"   进度: {current_point:,}/{total_points:,} ({progress:.2f}%) | "
                          f"可行解: {feasible_count:,} ({success_rate:.2f}%) | "
                          f"预计剩余: {est_time_left}")
        
        # 保存剩余结果
        if batch_results:
            for result in batch_results:
                self.save_result(*result)
        
        self.update_progress(total_points, current_point, feasible_count, start_time)
        
        print(f"\n✅ 分析完成!")
        print(f"   总点数: {total_points:,}")
        print(f"   可行解: {feasible_count:,}")
        print(f"   成功率: {100*feasible_count/total_points:.2f}%")
        print(f"   结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   负荷因子: {self.config['system']['load_scale_factor']:.2f}")
        
        # 分析完成后标记，方便下次自动清理
        print(f"💾 结果已保存到数据库，下次运行将自动清理")
    
    def estimate_time_left(self, completed, total, start_time):
        """估算剩余时间"""
        if completed == 0:
            return "计算中..."
        
        elapsed = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
        rate = completed / elapsed  # 点/秒
        remaining_points = total - completed
        remaining_seconds = remaining_points / rate
        
        if remaining_seconds < 60:
            return f"{remaining_seconds:.0f}秒"
        elif remaining_seconds < 3600:
            return f"{remaining_seconds/60:.0f}分钟"
        else:
            return f"{remaining_seconds/3600:.1f}小时"
    
    def plot_results(self, limit=None):
        """绘制结果"""
        # 获取分析配置信息
        cursor = self.conn.execute('SELECT * FROM analysis_config WHERE id = 1')
        config_data = cursor.fetchone()
        
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
        
        cbar = plt.colorbar(scatter, label='发电机3功率 P3 (MW)')
        cbar.ax.tick_params(labelsize=11)
        
        plt.xlabel('平衡机组功率 P1 (MW)', fontsize=14)
        plt.ylabel('发电机2功率 P2 (MW)', fontsize=14)
        
        # 改进的标题，包含负荷因子信息
        if config_data:
            load_factor = config_data[1]  # load_factor
            total_load = config_data[7]   # total_load
            title = f'负荷因子{load_factor:.2f}下的AC-OPF可行域\n(总负荷{total_load:.1f}MW，共{len(df):,}个可行解)'
        else:
            title = f'AC-OPF可行域 (共{len(df):,}个可行解)'
            
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 输出统计
        print(f"\n📈 绘图数据统计:")
        print(f"   可行解数量: {len(df):,}")
        print(f"   P1范围: {df['p1_mw'].min():.2f} - {df['p1_mw'].max():.2f} MW")
        print(f"   P2范围: {df['p2_mw'].min():.2f} - {df['p2_mw'].max():.2f} MW")
        print(f"   P3范围: {df['p3_mw'].min():.2f} - {df['p3_mw'].max():.2f} MW")
        
        if config_data:
            print(f"   负荷因子: {config_data[1]:.2f}") 
            print(f"   总负荷: {config_data[7]:.1f} MW")
    
    def get_statistics(self):
        """获取统计信息"""
        cursor = self.conn.execute('''
            SELECT COUNT(*) as total_feasible,
                   AVG(p1_mw) as avg_p1,
                   MIN(p1_mw) as min_p1,
                   MAX(p1_mw) as max_p1,
                   AVG(calculation_time) as avg_calc_time
            FROM feasible_solutions
        ''')
        stats = cursor.fetchone()
        
        cursor = self.conn.execute('SELECT * FROM progress WHERE id = 1')
        progress = cursor.fetchone()
        
        cursor = self.conn.execute('SELECT * FROM analysis_config WHERE id = 1')
        config_data = cursor.fetchone()
        
        return {
            'feasible_solutions': stats[0] if stats[0] else 0,
            'avg_p1': stats[1],
            'min_p1': stats[2], 
            'max_p1': stats[3],
            'avg_calc_time': stats[4],
            'total_points': progress[1] if progress else 0,
            'completed_points': progress[2] if progress else 0,
            'success_rate': 100 * stats[0] / progress[2] if progress and progress[2] > 0 else 0,
            'load_factor': config_data[1] if config_data else None,
            'total_load': config_data[7] if config_data else None
        }
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()

# 使用示例
if __name__ == "__main__":
    print("🔧 AC-OPF大规模可行域分析工具")
    print("="*50)
    
    # 创建分析器
    analyzer = ACOPFAnalyzer()
    
    try:
        # 显示当前配置
        print(f"📋 当前配置:")
        print(f"   网格大小: {analyzer.config['calculation']['n_points']}×{analyzer.config['calculation']['n_points']}")
        print(f"   负荷缩放因子: {analyzer.config['system']['load_scale_factor']}")
        print(f"   扫描范围: P2∈[{analyzer.p2_range[0]:.0f},{analyzer.p2_range[1]:.0f}], P3∈[{analyzer.p3_range[0]:.0f},{analyzer.p3_range[1]:.0f}]")  
        print(f"   自动清理: {analyzer.config['database'].get('auto_clear', True)}")
        
        # 菜单选择
        while True:
            print(f"\n📞 请选择操作:")
            print("1. 运行可行域分析")
            print("2. 绘制结果图")
            print("3. 查看统计信息")
            print("4. 修改配置")
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
                print(f"   总计算点: {stats['completed_points']:,}")
                print(f"   成功率: {stats['success_rate']:.2f}%")
                if stats['load_factor']:
                    print(f"   负荷因子: {stats['load_factor']:.2f}")
                    print(f"   总负荷: {stats['total_load']:.1f} MW")
                if stats['avg_p1']:
                    print(f"   P1平均: {stats['avg_p1']:.2f} MW")
                    print(f"   P1范围: {stats['min_p1']:.2f} - {stats['max_p1']:.2f} MW")
                    print(f"   平均计算时间: {stats['avg_calc_time']*1000:.2f} ms")
            elif choice == '4':
                print("💡 请编辑 ac_opf_config.json 文件修改配置")
                print("   修改后重新运行程序即可应用新配置")
            elif choice == '5':
                break
            else:
                print("❌ 无效选择")
    
    finally:
        analyzer.close()
        print("👋 程序结束")
