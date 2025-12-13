import re
import math
import time
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ========== 工具：解析 MATPOWER case .m 文件 ==========
def load_matpower_case(file_path: str):
    """
    轻量解析 MATPOWER .m 文件，提取 baseMVA、bus、gen、branch。
    支持数学表达式（如 135/sqrt(3)）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # 创建安全的数学命名空间
    math_namespace = {
        'sqrt': math.sqrt,
        'pi': math.pi,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'abs': abs,
        'pow': pow,
    }

    def safe_eval(expr):
        """安全地计算数学表达式"""
        try:
            return float(expr)
        except ValueError:
            try:
                return float(eval(expr, {"__builtins__": {}}, math_namespace))
            except:
                raise ValueError(f"无法解析表达式: {expr}")

    def extract_block(name):
        pattern = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
        m = re.search(pattern, txt, flags=re.S | re.M)
        if not m:
            raise ValueError(f"无法在 {file_path} 中找到 {name} 数据块")
        block = m.group(1).strip()
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if '%' in line:
                line = line.split('%', 1)[0].strip()
            if not line:
                continue
            line = line.rstrip(';').strip()
            if not line:
                continue
            tokens = re.split(r"[\s\t]+", line)
            nums = [safe_eval(tok) for tok in tokens if tok]
            if nums:
                rows.append(nums)
        return np.array(rows, dtype=float)

    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9.+\-Ee/]+)\s*;", txt)
    if not m:
        raise ValueError("未找到 mpc.baseMVA")
    baseMVA_expr = m.group(1)
    baseMVA = eval(baseMVA_expr)  # 处理 50/3 这样的表达式

    bus = extract_block('bus')
    gen = extract_block('gen')
    branch = extract_block('branch')

    return {
        'baseMVA': baseMVA,
        'bus': bus,
        'gen': gen,
        'branch': branch
    }

# ========== case533 单机组 AC-OPF 可行性扫描分析器 ==========
class ACOPF533SingleGenAnalyzer:
    def __init__(self,
                 case_file='case533mt_lo.m',
                 n_points=20,
                 gen_bus=None,
                 range_frac=(-0.30, 1.00),  # 发电机出力范围（相对 Pmax，可为负）
                 solver_opts=None):
        """
        case533 单台发电机 AC-OPF 可行性扫描
        
        扫描唯一发电机在不同有功出力水平下的可行性。
        gen_bus: 扫描的发电机所在母线号（默认自动选择）
        range_frac: 扫描相对范围（相对于 Pmax），可为负值表示吸收功率
        """
        data = load_matpower_case(case_file)
        self.baseMVA = data['baseMVA']
        self.bus = data['bus']
        self.gen = data['gen']
        self.branch = data['branch']

        self.n_bus = self.bus.shape[0]
        self.n_gen = self.gen.shape[0]
        self.n_branch = self.branch.shape[0]

        self.bus_i = self.bus[:, 0].astype(int)
        self.bus_type = self.bus[:, 1].astype(int)
        self.Pd = self.bus[:, 2] / self.baseMVA
        self.Qd = self.bus[:, 3] / self.baseMVA
        self.Gs = self.bus[:, 4] / self.baseMVA
        self.Bs = self.bus[:, 5] / self.baseMVA
        self.Vm0 = self.bus[:, 7]
        self.Va0_deg = self.bus[:, 8]
        self.Vmax = self.bus[:, 11]
        self.Vmin = self.bus[:, 12]

        self.gen_bus = self.gen[:, 0].astype(int)
        self.gen_status = self.gen[:, 7].astype(int)
        self.Pmax = self.gen[:, 8] / self.baseMVA
        self.Pmin = self.gen[:, 9] / self.baseMVA
        self.Qmax = self.gen[:, 3] / self.baseMVA
        self.Qmin = self.gen[:, 4] / self.baseMVA
        self.Vg_set = self.gen[:, 5]

        self.active_gen_idx = [i for i in range(self.n_gen) if self.gen_status[i] == 1]

        if len(self.active_gen_idx) == 0:
            raise ValueError("没有在运发电机！")

        self.bus_to_gens = {int(b): [] for b in self.bus_i}
        for gi in self.active_gen_idx:
            self.bus_to_gens[int(self.gen_bus[gi])].append(gi)

        slack_candidates = self.bus_i[self.bus_type == 3]
        if len(slack_candidates) == 0:
            raise ValueError("未找到参考母线 (bus type == 3)")
        self.slack_bus = int(slack_candidates[0])

        self.busnum_to_idx = {int(self.bus_i[k]): k for k in range(self.n_bus)}

        self.Ybus = self.build_Ybus()
        self.G = self.Ybus.real
        self.B = self.Ybus.imag

        # 选择扫描机组
        if gen_bus is None:
            self.gen_idx = self.active_gen_idx[0]
            self.gen_bus_num = int(self.gen_bus[self.gen_idx])
        else:
            self.gen_bus_num = int(gen_bus)
            self.gen_idx = self.find_active_gen_index_by_bus(self.gen_bus_num)

        # 扫描范围
        lo, hi = range_frac
        self.pg_min_pu = max(self.Pmin[self.gen_idx], lo * self.Pmax[self.gen_idx])
        self.pg_max_pu = min(self.Pmax[self.gen_idx], hi * self.Pmax[self.gen_idx])

        self.n_points = int(n_points)
        
        self.solver_opts = solver_opts or {
            'max_iter': 400,
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'print_level': 0,
            'linear_solver': 'mumps'
        }

        self.feasible_times = []
        self.avg100_reported = False

        # 打印信息
        print("===== case533 单机组 AC-OPF 可行性扫描 =====")
        print(f"BaseMVA: {self.baseMVA:.2f}")
        print(f"母线数: {self.n_bus}, 发电机数: {self.n_gen}, 支路数: {self.n_branch}")
        print(f"参考母线/源点: {self.slack_bus}")
        print(f"电压等级: 135/12 kV（配电网）")
        total_Pd = np.sum(self.bus[:, 2])
        total_Qd = np.sum(self.bus[:, 3])
        print(f"系统净负荷: {total_Pd:.2f} MW, {total_Qd:.2f} MVar (负值表示DG注入大于负荷)")
        
        print(f"\n扫描发电机: gen#{self.gen_idx+1} @ bus {self.gen_bus_num}")
        print(f"  Pmax = {self.Pmax[self.gen_idx]*self.baseMVA:.2f} MW")
        print(f"  Pmin = {self.Pmin[self.gen_idx]*self.baseMVA:.2f} MW")
        print(f"  Qmax = {self.Qmax[self.gen_idx]*self.baseMVA:.2f} MVar")
        print(f"  Qmin = {self.Qmin[self.gen_idx]*self.baseMVA:.2f} MVar")
        print(f"\n扫描有功出力范围（MW）: [{self.pg_min_pu*self.baseMVA:.2f}, {self.pg_max_pu*self.baseMVA:.2f}]")
        print(f"  (负值表示从电网吸收功率)")
        print(f"扫描点数: {self.n_points}\n")

    def find_active_gen_index_by_bus(self, bus_no: int):
        idxs = [i for i in self.active_gen_idx if int(self.gen_bus[i]) == int(bus_no)]
        if not idxs:
            raise ValueError(f"未找到位于母线 {bus_no} 的在运发电机")
        return idxs[0]

    def build_Ybus(self):
        nb = self.n_bus
        Y = np.zeros((nb, nb), dtype=complex)

        for i in range(nb):
            Y[i, i] += complex(self.Gs[i], self.Bs[i])

        for k in range(self.n_branch):
            fbus = int(self.branch[k, 0])
            tbus = int(self.branch[k, 1])
            r, x, b = self.branch[k, 2:5]
            ratio = self.branch[k, 8]
            angle_deg = self.branch[k, 9]
            status = int(self.branch[k, 10])
            if status == 0:
                continue

            f = self.busnum_to_idx[fbus]
            t = self.busnum_to_idx[tbus]

            z = complex(r, x)
            if abs(z) < 1e-12:
                y = 1 / (1j * x) if abs(x) > 1e-12 else 1e12
            else:
                y = 1 / z
            jb = 1j * (b / 2.0)

            tap = ratio if ratio != 0 else 1.0
            shift_rad = math.radians(angle_deg)
            t_complex = tap * complex(math.cos(shift_rad), math.sin(shift_rad))

            Y[f, f] += (y + jb) / (t_complex * np.conj(t_complex))
            Y[t, t] += (y + jb)
            Y[f, t] += - y / np.conj(t_complex)
            Y[t, f] += - y / (t_complex)

        return Y

    def solve_feasibility(self, pg_pu):
        """
        给定发电机有功出力（标幺），求解 AC 可行性（目标为常数 0）
        无功 Qg 自由优化
        """
        t0 = time.time()
        try:
            m = pyo.ConcreteModel()
            m.BUS = pyo.RangeSet(1, self.n_bus)
            m.GEN = pyo.RangeSet(1, self.n_gen)

            m.Vm = pyo.Var(m.BUS, bounds=lambda _, i: (self.Vmin[i-1], self.Vmax[i-1]), initialize=1.0)
            m.Va = pyo.Var(m.BUS, bounds=(-math.pi, math.pi), initialize=0.0)

            def pg_bounds(_, g):
                gi = g - 1
                return (0.0, 0.0) if self.gen_status[gi] == 0 else (self.Pmin[gi], self.Pmax[gi])

            def qg_bounds(_, g):
                gi = g - 1
                return (0.0, 0.0) if self.gen_status[gi] == 0 else (self.Qmin[gi], self.Qmax[gi])

            m.Pg = pyo.Var(m.GEN, bounds=pg_bounds)
            m.Qg = pyo.Var(m.GEN, bounds=qg_bounds)

            for g in m.GEN:
                gi = g - 1
                if self.gen_status[gi] == 0:
                    m.Pg[g].fix(0.0)
                    m.Qg[g].fix(0.0)
                else:
                    pg0 = float(self.gen[gi, 1]) / self.baseMVA
                    qg0 = float(self.gen[gi, 2]) / self.baseMVA
                    pg0 = min(max(pg0, self.Pmin[gi]), self.Pmax[gi])
                    qg0 = min(max(qg0, self.Qmin[gi]), self.Qmax[gi])
                    m.Pg[g].value = pg0
                    m.Qg[g].value = qg0

            # 固定扫描机组的 Pg
            m.Pg[self.gen_idx + 1].fix(pg_pu)

            slack_pos = self.busnum_to_idx[self.slack_bus] + 1
            m.ref = pyo.Constraint(expr=m.Va[slack_pos] == 0.0)

            m.OBJ = pyo.Objective(expr=0.0, sense=pyo.minimize)

            G, B = self.G, self.B

            def P_balance_rule(_m, i):
                idx = i - 1
                Pi = 0.0
                for j in _m.BUS:
                    gij = G[idx, j-1]; bij = B[idx, j-1]
                    Pi += _m.Vm[i]*_m.Vm[j]*(gij*pyo.cos(_m.Va[i]-_m.Va[j]) + bij*pyo.sin(_m.Va[i]-_m.Va[j]))
                pg_sum = 0.0
                bus_no = int(self.bus_i[idx])
                for gi in self.bus_to_gens.get(bus_no, []):
                    pg_sum += _m.Pg[gi + 1]
                return pg_sum - self.Pd[idx] == Pi

            def Q_balance_rule(_m, i):
                idx = i - 1
                Qi = 0.0
                for j in _m.BUS:
                    gij = G[idx, j-1]; bij = B[idx, j-1]
                    Qi += _m.Vm[i]*_m.Vm[j]*(gij*pyo.sin(_m.Va[i]-_m.Va[j]) - bij*pyo.cos(_m.Va[i]-_m.Va[j]))
                qg_sum = 0.0
                bus_no = int(self.bus_i[idx])
                for gi in self.bus_to_gens.get(bus_no, []):
                    qg_sum += _m.Qg[gi + 1]
                return qg_sum - self.Qd[idx] == Qi

            m.Pbal = pyo.Constraint(m.BUS, rule=P_balance_rule)
            m.Qbal = pyo.Constraint(m.BUS, rule=Q_balance_rule)

            for i in m.BUS:
                bus_idx = i - 1
                v0 = min(max(self.Vm0[bus_idx], self.Vmin[bus_idx]), self.Vmax[bus_idx])
                m.Vm[i].value = v0
                m.Va[i].value = math.radians(self.Va0_deg[bus_idx])
            for gi in self.active_gen_idx:
                bno = int(self.gen_bus[gi])
                i = self.busnum_to_idx[bno] + 1
                vg = min(max(float(self.Vg_set[gi]), self.Vmin[i-1]), self.Vmax[i-1])
                m.Vm[i].value = vg

            solver = SolverFactory('ipopt')
            for k, v in self.solver_opts.items():
                solver.options[k] = v
            res = solver.solve(m, tee=False)
            t_used = time.time() - t0

            term = res.solver.termination_condition
            feasible = term in (pyo.TerminationCondition.optimal,
                                pyo.TerminationCondition.locallyOptimal,
                                pyo.TerminationCondition.feasible)
            if not feasible:
                return False, None, t_used

            def safe_val(x):
                try:
                    return float(pyo.value(x))
                except Exception:
                    return 0.0

            sol = {
                'Pg_MW': [safe_val(m.Pg[g]) * self.baseMVA for g in m.GEN],
                'Qg_MVar': [safe_val(m.Qg[g]) * self.baseMVA for g in m.GEN],
                'Vm': [safe_val(m.Vm[i]) for i in m.BUS],
                'Va_deg': [safe_val(m.Va[i]) * 180.0 / math.pi for i in m.BUS]
            }
            return True, sol, t_used

        except Exception:
            return False, None, time.time() - t0

    def run_scan(self):
        """
        一维扫描：Pg(gen) 从 pg_min 到 pg_max
        - 打印每个可行解的时间
        - 累计到100个可行解时，计算并打印前100个平均时间（仅一次）
        - 扫描结束打印整体平均时间
        """
        print("开始扫描 ...")
        pg_vals_pu = np.linspace(self.pg_min_pu, self.pg_max_pu, self.n_points)

        feasible_count = 0
        total_points = len(pg_vals_pu)
        processed = 0

        for pg in pg_vals_pu:
            processed += 1
            ok, sol, tsec = self.solve_feasibility(pg)
            if ok:
                feasible_count += 1
                self.feasible_times.append(tsec)
                qg = sol['Qg_MVar'][self.gen_idx]
                print(f"✅ 可行解 #{feasible_count:>4}: "
                      f"Pg={pg*self.baseMVA:.2f} MW, "
                      f"Qg={qg:.2f} MVar, "
                      f"用时={tsec:.3f}s")
                if (not self.avg100_reported) and len(self.feasible_times) >= 100:
                    avg100 = sum(self.feasible_times[:100]) / 100.0
                    self.avg100_reported = True
                    print(f"🎯 前100个可行解平均求解时间：{avg100:.3f}s")
            
            if processed % max(1, total_points // 10) == 0 or processed == total_points:
                prog = 100.0 * processed / total_points
                print(f"进度: {processed}/{total_points} ({prog:.1f}%) | 已获可行解: {feasible_count}")

        if self.feasible_times:
            overall_avg = sum(self.feasible_times) / len(self.feasible_times)
            print(f"\n⏱️ 本次共获得 {len(self.feasible_times)} 个可行解；整体平均求解时间 = {overall_avg:.3f}s")
        else:
            print("\n未获得可行解。可考虑：微调扫描范围、或调整求解器参数。")

# ========== 运行示例 ==========
if __name__ == "__main__":
    analyzer = ACOPF533SingleGenAnalyzer(
        case_file='case533mt_lo.m',
        n_points=200,                      # 20个出力水平点
        gen_bus=None,                     # 自动选择第一台在运机组（bus 1）
        range_frac=(-0.30, 1.00),         # 扫描 -30% 到 100% Pmax
        solver_opts={
            'max_iter': 400,
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'print_level': 0,
            'linear_solver': 'mumps'
        }
    )
    analyzer.run_scan()