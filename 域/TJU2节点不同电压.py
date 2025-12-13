import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
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

class TwoNodeACOPFAnalyzer:
    def __init__(self, config_file='two_node_ac_opf_config.json'):
        """初始化2节点AC-OPF分析器"""
        self.config = self.load_config(config_file)
        self.setup_system_data()
        
    def load_config(self, config_file):
        """加载配置文件"""
        default_config = {
            "system": {
                "baseMVA": 100,
                "thermal_limit_factor": 1.0,
            },
            "calculation": {
                "n_points": 100,  # 减少点数以提高稳定性
                "batch_size": 50,
                "save_interval": 25
            },
            "solver": {
                "max_iter": 500,
                "tol": 1e-5,
                "acceptable_tol": 1e-4,
                "print_level": 0,
                "linear_solver": "ma57"  # 使用更稳定的线性求解器
            },
            "debug": {
                "enabled": True,
                "verbose": False,
                "save_infeasible": True
            },
            "output": {
                "csv_filename": "two_node_ac_opf_results.csv",
                "plot_filename": "two_node_v1_v2_plot.png"
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
    
    def setup_system_data(self):
        """设置2节点系统数据"""
        self.baseMVA = self.config['system']['baseMVA']
        thermal_factor = self.config['system']['thermal_limit_factor']
        
        # 2节点系统数据
        self.bus = np.array([
            [1, 3, 0,   0,   0, 0, 1, 0.964, 0,   0, 1, 1.05, 0.95],  # 平衡节点
            [2, 1, 350, -350, 0, 0, 1, 1.0,   -65, 0, 1, 1.05, 0.95]   # PQ节点
        ])
        
        # 发电机数据
        self.gen = np.array([
            [1, 400, 100, 400, -400, 0.964, 100, 1, 600, 0]  # 发电机在节点1
        ])
        
        # 支路数据
        branch_base = np.array([
            [1, 2, 0.04, 0.2, 0, 990000, 0, 0, 0, 0, 1, -360, 360]
        ])
        
        # 应用热稳定限值缩放
        self.branch = branch_base.copy()
        self.branch[:, 5] = branch_base[:, 5] * thermal_factor
        
        self.setup_system_matrices()
        
        # 设置P1扫描范围（基于发电机约束）
        self.p1_range = [420, 480]  # 使用您指定的范围
        
        total_load = sum(self.bus[:, 2])
        print(f"⚡ 2节点系统配置:")
        print(f"   基准功率: {self.baseMVA} MVA")
        print(f"   总有功负荷: {total_load:.1f} MW")
        print(f"   线路容量缩放因子: {thermal_factor:.2f}")
        print(f"   发电机约束:")
        print(f"     P1∈[{self.p1_range[0]:.0f},{self.p1_range[1]:.0f}]MW")
        print(f"   电压约束:")
        print(f"     V1∈[{self.Vmin[0]:.3f},{self.Vmax[0]:.3f}] p.u.")
        print(f"     V2∈[{self.Vmin[1]:.3f},{self.Vmax[1]:.3f}] p.u.")
    
    def setup_system_matrices(self):
        """建立2节点系统矩阵"""
        self.n_bus = self.bus.shape[0]
        self.n_gen = self.gen.shape[0]
        self.n_branch = self.branch.shape[0]
        
        # 发电机母线映射
        self.gen_buses = [int(self.gen[i, 0]) - 1 for i in range(self.n_gen)]
        
        # 负荷数据（标幺值）
        self.Pd = [float(self.bus[i, 2] / self.baseMVA) for i in range(self.n_bus)]
        self.Qd = [float(self.bus[i, 3] / self.baseMVA) for i in range(self.n_bus)]
        
        # 电压约束
        self.Vmax = [float(self.bus[i, 11]) for i in range(self.n_bus)]
        self.Vmin = [float(self.bus[i, 12]) for i in range(self.n_bus)]
        
        # 发电机约束（标幺值）
        self.gen_pmax = [float(self.gen[i, 8] / self.baseMVA) for i in range(self.n_gen)]
        self.gen_pmin = [float(self.gen[i, 9] / self.baseMVA) for i in range(self.n_gen)]
        self.gen_qmax = [float(self.gen[i, 3] / self.baseMVA) for i in range(self.n_gen)]
        self.gen_qmin = [float(self.gen[i, 4] / self.baseMVA) for i in range(self.n_gen)]
        
        # 发电机映射
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
        
        # 参考节点（平衡节点）
        self.swing_bus = 1
    
    def build_opf_model(self, p1_pu):
        """构建2节点AC-OPF模型"""
        model = pyo.ConcreteModel()
        
        # 定义集合
        model.buses = pyo.Set(initialize=[1, 2])
        model.gens = pyo.Set(initialize=[1])
        model.branches = pyo.Set(initialize=[1])
        
        # 定义变量
        model.Vm = pyo.Var(model.buses, bounds=lambda m, i: (self.Vmin[i-1], self.Vmax[i-1]), initialize=1.0)
        model.Va = pyo.Var(model.buses, bounds=(-math.pi/4, math.pi/4), initialize=0.0)
        
        model.Pg = pyo.Var(model.gens, bounds=lambda m, g: (self.gen_pmin[g-1], self.gen_pmax[g-1]))
        model.Qg = pyo.Var(model.gens, bounds=lambda m, g: (self.gen_qmin[g-1], self.gen_qmax[g-1]), initialize=0.0)
        
        # 目标函数（最小化成本，这里设为0）
        model.cost = pyo.Objective(expr=0, sense=pyo.minimize)
        
        # 固定P1
        model.pg1_fixed = pyo.Constraint(expr=model.Pg[1] == p1_pu)
        
        # 功率平衡约束
        def power_balance_p(m, i):
            Pi = 0.0
            for j in m.buses:
                gij = self.G[i-1][j-1]
                bij = self.B[i-1][j-1]
                Pi += m.Vm[i] * m.Vm[j] * (gij * pyo.cos(m.Va[i] - m.Va[j]) + bij * pyo.sin(m.Va[i] - m.Va[j]))
            
            # 节点1有发电机，节点2没有发电机
            if i == 1:  # 节点1
                Pg_inject = m.Pg[1]
            else:  # 节点2
                Pg_inject = 0
                
            return Pg_inject - self.Pd[i-1] == Pi
        
        def power_balance_q(m, i):
            Qi = 0.0
            for j in m.buses:
                gij = self.G[i-1][j-1]
                bij = self.B[i-1][j-1]
                Qi += m.Vm[i] * m.Vm[j] * (gij * pyo.sin(m.Va[i] - m.Va[j]) - bij * pyo.cos(m.Va[i] - m.Va[j]))
            
            # 节点1有发电机，节点2没有发电机
            if i == 1:  # 节点1
                Qg_inject = m.Qg[1]
            else:  # 节点2
                Qg_inject = 0
                
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
            y = 1/complex(r, x) if abs(complex(r, x)) > 1e-8 else 1/complex(0, x)
            g = y.real
            b_y = y.imag
            
            # 有功潮流
            Pf = g * Vf**2 - Vf * Vt * (g * pyo.cos(theta_ft) + b_y * pyo.sin(theta_ft))
            
            # 热稳定约束
            thermal_limit = br[5] / self.baseMVA
            return Pf**2 <= thermal_limit**2
        
        model.thermal_limits = pyo.Constraint(model.branches, rule=branch_flow_rule)
        
        return model
    
    def solve_feasibility(self, p1_pu):
        """求解单个P1点的可行性"""
        start_time = time.time()
        
        try:
            # 重定向输出到字符串，避免控制台输出
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            # 构建模型
            model = self.build_opf_model(p1_pu)
            
            # 求解器设置 - 使用更稳定的配置
            solver = SolverFactory('ipopt')
            solver.options['max_iter'] = self.config['solver']['max_iter']
            solver.options['tol'] = self.config['solver']['tol']
            solver.options['acceptable_tol'] = self.config['solver']['acceptable_tol']
            solver.options['print_level'] = self.config['solver']['print_level']
            
            # 添加稳定性选项
            solver.options['linear_solver'] = self.config['solver']['linear_solver']
            solver.options['mu_strategy'] = 'adaptive'
            solver.options['bound_frac'] = 0.001
            solver.options['bound_push'] = 0.001
            solver.options['corrector_type'] = 'affine'
            solver.options['expect_infeasible_problem'] = 'no'
            
            # 求解
            results = solver.solve(model, tee=False)
            
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            calc_time = time.time() - start_time
            
            # 检查求解结果
            if str(results.solver.termination_condition) in ['optimal', 'locallyOptimal', 'feasible']:
                # 提取所有变量值
                solution_data = {}
                
                # 发电机有功无功
                solution_data['p1_mw'] = pyo.value(model.Pg[1]) * self.baseMVA
                solution_data['q1_mvar'] = pyo.value(model.Qg[1]) * self.baseMVA
                
                # 电压幅值和相角
                for i in range(1, 3):  # 只有2个节点
                    solution_data[f'v{i}_pu'] = pyo.value(model.Vm[i])
                    solution_data[f'theta{i}_deg'] = pyo.value(model.Va[i]) * 180 / math.pi
                
                return True, solution_data, calc_time
            else:
                return False, None, calc_time
                
        except Exception as e:
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print(f"⚠️ 求解出错: {e}")
            return False, None, time.time() - start_time
    
    def run_analysis(self):
        """运行2节点AC-OPF可行域分析"""
        print(f"\n🚀 开始2节点AC-OPF可行域分析")
        print("="*60)
        
        n_points = self.config['calculation']['n_points']
        save_interval = self.config['calculation']['save_interval']
        
        # 生成P1扫描点
        p1_values = np.linspace(self.p1_range[0], self.p1_range[1], n_points)
        
        total_points = len(p1_values)
        
        print(f"📊 计算规模:")
        print(f"   扫描点数: {total_points:,}")
        print(f"   P1扫描范围: {self.p1_range[0]:.1f} - {self.p1_range[1]:.1f} MW")
        print(f"   线路容量缩放因子: {self.config['system']['thermal_limit_factor']:.2f}")
        
        start_time = time.time()
        feasible_results = []
        all_results = []
        
        print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔄 计算进行中...")
        
        for i, p1_mw in enumerate(p1_values):
            # 转换为标幺值并求解
            p1_pu = p1_mw / self.baseMVA
            
            is_feasible, solution_data, calc_time = self.solve_feasibility(p1_pu)
            
            result = {
                'p1_mw': p1_mw,
                'is_feasible': is_feasible,
                'calculation_time': calc_time,
                'timestamp': datetime.now().isoformat()
            }
            
            if is_feasible and solution_data:
                result.update(solution_data)
                feasible_results.append(result)
                status = "可行"
            else:
                status = "不可行"
            
            all_results.append(result)
            
            # 定期更新进度
            if (i + 1) % save_interval == 0 or (i + 1) == total_points:
                progress = 100 * (i + 1) / total_points
                success_rate = 100 * len(feasible_results) / (i + 1) if (i + 1) > 0 else 0
                elapsed_time = time.time() - start_time
                
                print(f"   进度: {i+1:,}/{total_points:,} ({progress:.1f}%) | "
                      f"可行解: {len(feasible_results):,} ({success_rate:.1f}%) | "
                      f"用时: {elapsed_time:.1f}秒")
        
        # 保存结果到CSV
        self.save_to_csv(all_results)
        
        print(f"\n✅ 分析完成!")
        print(f"   总点数: {total_points:,}")
        print(f"   可行解: {len(feasible_results):,}")
        print(f"   成功率: {100*len(feasible_results)/total_points:.1f}%")
        print(f"   总用时: {time.time() - start_time:.1f}秒")
        print(f"   结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return feasible_results, all_results
    
    def save_to_csv(self, results):
        """保存结果到CSV文件"""
        # 创建DataFrame
        df_data = []
        for result in results:
            row = {
                'P1_MW': result['p1_mw'],
                'Is_Feasible': result['is_feasible'],
                'Calculation_Time_s': result['calculation_time'],
                'Timestamp': result['timestamp']
            }
            
            if result['is_feasible']:
                # 添加电压和相角数据
                for i in range(1, 3):  # 只有2个节点
                    row[f'V{i}_pu'] = result.get(f'v{i}_pu', None)
                    row[f'Theta{i}_deg'] = result.get(f'theta{i}_deg', None)
                
                # 添加发电机数据
                row['Q1_MVAR'] = result.get('q1_mvar', None)
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_filename = self.config['output']['csv_filename']
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"💾 结果已保存到: {csv_filename}")
        
        # 同时保存可行解到单独文件
        feasible_df = df[df['Is_Feasible'] == True]
        if len(feasible_df) > 0:
            feasible_filename = csv_filename.replace('.csv', '_feasible.csv')
            feasible_df.to_csv(feasible_filename, index=False, encoding='utf-8-sig')
            print(f"💾 可行解已保存到: {feasible_filename}")
    
    def plot_results(self, results=None):
        """绘制V1-V2散点图"""
        if results is None:
            # 从CSV文件加载数据
            csv_filename = self.config['output']['csv_filename']
            if not os.path.exists(csv_filename):
                print("❌ 未找到结果文件，请先运行分析")
                return
            
            df = pd.read_csv(csv_filename)
            feasible_df = df[df['Is_Feasible'] == True]
        else:
            # 使用提供的结果数据
            feasible_results = [r for r in results if r['is_feasible']]
            if len(feasible_results) == 0:
                print("❌ 没有可行解数据可用于绘图")
                return
            
            # 转换为DataFrame
            df_data = []
            for result in feasible_results:
                row = {'P1_MW': result['p1_mw']}
                for i in range(1, 3):  # 只有2个节点
                    row[f'V{i}_pu'] = result.get(f'v{i}_pu', None)
                df_data.append(row)
            
            feasible_df = pd.DataFrame(df_data)
        
        if len(feasible_df) == 0:
            print("❌ 没有可行解数据可用于绘图")
            return
        
        # 绘制V1-V2散点图
        plt.figure(figsize=(10, 8))
        
        # 提取V1和V2数据
        v1_values = feasible_df['V1_pu'].values
        v2_values = feasible_df['V2_pu'].values
        
        # 创建散点图
        plt.scatter(v1_values, v2_values, s=5, c='tab:blue', alpha=0.6, label='Feasible Region')
        plt.legend()
        
        # 设置坐标轴标签和标题
        plt.xlabel('V1 (p.u.)', fontsize=14)
        plt.ylabel('V2 (p.u.)', fontsize=14)
        
        thermal_factor = self.config['system']['thermal_limit_factor']
        title = f'2节点系统AC-OPF可行域 (线路容量因子: {thermal_factor:.2f})\n共{len(feasible_df)}个可行解'
        plt.title(title, fontsize=16, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plot_filename = self.config['output']['plot_filename']
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 图形已保存到: {plot_filename}")
        
        plt.show()
        
        # 输出统计信息
        print(f"\n📈 绘图数据统计:")
        print(f"   可行解数量: {len(feasible_df):,}")
        print(f"   V1范围: {feasible_df['V1_pu'].min():.4f} - {feasible_df['V1_pu'].max():.4f} p.u.")
        print(f"   V2范围: {feasible_df['V2_pu'].min():.4f} - {feasible_df['V2_pu'].max():.4f} p.u.")
        print(f"   P1范围: {feasible_df['P1_MW'].min():.1f} - {feasible_df['P1_MW'].max():.1f} MW")
    
    def get_statistics(self):
        """获取统计信息"""
        csv_filename = self.config['output']['csv_filename']
        if not os.path.exists(csv_filename):
            return {"error": "结果文件不存在"}
        
        df = pd.read_csv(csv_filename)
        total_points = len(df)
        feasible_points = len(df[df['Is_Feasible'] == True])
        
        stats = {
            'total_points': total_points,
            'feasible_points': feasible_points,
            'success_rate': 100 * feasible_points / total_points if total_points > 0 else 0,
        }
        
        if feasible_points > 0:
            feasible_df = df[df['Is_Feasible'] == True]
            stats.update({
                'v1_min': feasible_df['V1_pu'].min(),
                'v1_max': feasible_df['V1_pu'].max(),
                'v2_min': feasible_df['V2_pu'].min(),
                'v2_max': feasible_df['V2_pu'].max(),
                'p1_min': feasible_df['P1_MW'].min(),
                'p1_max': feasible_df['P1_MW'].max(),
            })
        
        return stats

# 使用示例
if __name__ == "__main__":
    print("🔧 2节点AC-OPF可行域分析工具")
    print("="*55)
    
    analyzer = TwoNodeACOPFAnalyzer()
    
    try:
        # 显示当前配置
        print(f"📋 当前配置:")
        print(f"   扫描点数: {analyzer.config['calculation']['n_points']}")
        print(f"   线路容量缩放因子: {analyzer.config['system']['thermal_limit_factor']}")
        print(f"   P1扫描范围: {analyzer.p1_range[0]:.0f} - {analyzer.p1_range[1]:.0f} MW")  
        
        # 菜单选择
        while True:
            print(f"\n📞 请选择操作:")
            print("1. 运行可行域分析")
            print("2. 绘制V1-V2散点图")
            print("3. 查看统计信息")
            print("4. 退出")
            
            choice = input("请输入选择 (1-4): ").strip()
            
            if choice == '1':
                feasible_results, all_results = analyzer.run_analysis()
                analyzer.plot_results(all_results)
            elif choice == '2':
                analyzer.plot_results()
            elif choice == '3':
                stats = analyzer.get_statistics()
                if 'error' in stats:
                    print(f"❌ {stats['error']}")
                else:
                    print(f"\n📊 统计信息:")
                    print(f"   总计算点: {stats['total_points']:,}")
                    print(f"   可行解: {stats['feasible_points']:,}")
                    print(f"   成功率: {stats['success_rate']:.1f}%")
                    
                    if stats['feasible_points'] > 0:
                        print(f"   V1范围: {stats['v1_min']:.4f} - {stats['v2_max']:.4f} p.u.")
                        print(f"   V2范围: {stats['v2_min']:.4f} - {stats['v2_max']:.4f} p.u.")
                        print(f"   P1范围: {stats['p1_min']:.1f} - {stats['p1_max']:.1f} MW")
            elif choice == '4':
                break
            else:
                print("❌ 无效选择")
    
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("👋 程序结束")