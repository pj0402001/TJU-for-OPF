import re
import math
import time
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ========== 工具：解析 MATPOWER case118.m ==========
def load_matpower_case(file_path: str):
    """
    轻量解析 MATPOWER .m 文件，提取 baseMVA、bus、gen、branch。
    要求文件中存在如下块：
      mpc.baseMVA = <num>;
      mpc.bus = [ ... ];
      mpc.gen = [ ... ];
      mpc.branch = [ ... ];
    返回: dict(baseMVA, bus(np.ndarray), gen(np.ndarray), branch(np.ndarray))
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    def extract_block(name):
        # 提取 mpc.<name> = [ ... ];
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
            # 去除行尾注释
            if '%' in line:
                line = line.split('%', 1)[0].strip()
            if not line:
                continue
            # 每行可能以 ; 结尾
            line = line.rstrip(';').strip()
            if not line:
                continue
            # 拆分数字
            tokens = re.split(r"[\s\t]+", line)
            nums = [float(tok) for tok in tokens if tok]
            if nums:
                rows.append(nums)
        return np.array(rows, dtype=float)

    # baseMVA
    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9.+\-Ee]+)\s*;", txt)
    if not m:
        raise ValueError("未找到 mpc.baseMVA")
    baseMVA = float(m.group(1))

    bus = extract_block('bus')
    gen = extract_block('gen')
    branch = extract_block('branch')

    return {
        'baseMVA': baseMVA,
        'bus': bus,
        'gen': gen,
        'branch': branch
    }

# ========== ACOPF 118-bus 分析器 ==========
class ACOPF118Analyzer:
    def __init__(self, case_file='case118.m', genA_bus=65, genB_bus=66, n_points=15, solver_opts=None):
        """
        case_file: MATPOWER case118.m 路径
        genA_bus, genB_bus: 扫描的两台发电机所在母线号（建议：65 与 66，容量充裕）
        n_points: 每个维度扫描点数
        solver_opts: Ipopt 选项字典
        """
        data = load_matpower_case(case_file)
        self.baseMVA = data['baseMVA']
        self.bus = data['bus']
        self.gen = data['gen']
        self.branch = data['branch']

        # 基本维度
        self.n_bus = self.bus.shape[0]
        self.n_gen = self.gen.shape[0]
        self.n_branch = self.branch.shape[0]

        # 索引与数据准备
        self.bus_i = self.bus[:, 0].astype(int)
        self.bus_type = self.bus[:, 1].astype(int)
        self.Pd = self.bus[:, 2] / self.baseMVA
        self.Qd = self.bus[:, 3] / self.baseMVA
        self.Gs = self.bus[:, 4] / self.baseMVA  # 并联有功导纳（注：MATPOWER中单位为 MW 在 1pu电压下）
        self.Bs = self.bus[:, 5] / self.baseMVA  # 并联无功电纳（MVAr）
        self.Vm0 = self.bus[:, 7]
        self.Va0_deg = self.bus[:, 8]
        self.Vmax = self.bus[:, 11]
        self.Vmin = self.bus[:, 12]

        # 发电机数据：列参考 MATPOWER
        # gen: [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, ...]
        self.gen_bus = self.gen[:, 0].astype(int)
        self.Pmax = self.gen[:, 8] / self.baseMVA
        self.Pmin = self.gen[:, 9] / self.baseMVA
        self.Qmax = self.gen[:, 3] / self.baseMVA
        self.Qmin = self.gen[:, 4] / self.baseMVA

        # 构建母线->发电机列表映射
        self.bus_to_gens = {b: [] for b in self.bus_i}
        for gi in range(self.n_gen):
            self.bus_to_gens[int(self.gen_bus[gi])].append(gi)

        # 参考母线（type==3）
        slack_candidates = self.bus_i[self.bus_type == 3]
        if len(slack_candidates) == 0:
            raise ValueError("未找到参考母线(type=3)")
        self.slack_bus = int(slack_candidates[0])

        # Ybus
        self.Ybus = self.build_Ybus()
        self.G = self.Ybus.real
        self.B = self.Ybus.imag

        # 选择两台扫描机组（通过母线号）
        self.genA_bus = genA_bus
        self.genB_bus = genB_bus
        self.genA_idx = self.find_gen_index_by_bus(genA_bus)
        self.genB_idx = self.find_gen_index_by_bus(genB_bus)

        # 自动选择“合适”的扫描范围（25% ~ 75% Pmax）
        self.pA_min_pu = max(self.Pmin[self.genA_idx], 0.25 * self.Pmax[self.genA_idx])
        self.pA_max_pu = 0.75 * self.Pmax[self.genA_idx]
        self.pB_min_pu = max(self.Pmin[self.genB_idx], 0.25 * self.Pmax[self.genB_idx])
        self.pB_max_pu = 0.75 * self.Pmax[self.genB_idx]

        self.n_points = n_points
        self.solver_opts = solver_opts or {
            'max_iter': 300,
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'print_level': 0
        }

        # 统计：可行解用时
        self.feasible_times = []
        self.avg100_reported = False

        print("===== IEEE 118 节点 AC-OPF 可行性扫描 =====")
        print(f"BaseMVA: {self.baseMVA}")
        print(f"母线数: {self.n_bus}, 发电机数: {self.n_gen}, 支路数: {self.n_branch}")
        print(f"参考母线: {self.slack_bus}")
        total_Pd = np.sum(self.bus[:, 2])
        print(f"系统总有功负荷: {total_Pd:.1f} MW")
        print(f"扫描机组A: bus {self.genA_bus} (Pmax={self.Pmax[self.genA_idx]*self.baseMVA:.1f} MW)")
        print(f"扫描机组B: bus {self.genB_bus} (Pmax={self.Pmax[self.genB_idx]*self.baseMVA:.1f} MW)")
        print("建议扫描范围（MW）:")
        print(f"  A: [{self.pA_min_pu*self.baseMVA:.1f}, {self.pA_max_pu*self.baseMVA:.1f}]")
        print(f"  B: [{self.pB_min_pu*self.baseMVA:.1f}, {self.pB_max_pu*self.baseMVA:.1f}]")

    def find_gen_index_by_bus(self, bus_no: int):
        idxs = np.where(self.gen_bus == bus_no)[0]
        if len(idxs) == 0:
            raise ValueError(f"未找到位于母线 {bus_no} 的发电机")
        return int(idxs[0])

    def build_Ybus(self):
        """
        构建带变比/相移的 Ybus 矩阵；包含线路π模型与母线并联导纳 Gs+jBs。
        MATPOWER branch 列：
          fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
        ratio=0 表示无变比；否则表示 t，angle 为度。
        """
        nb = self.n_bus
        Y = np.zeros((nb, nb), dtype=complex)

        # 加入母线并联导纳
        for i in range(nb):
            Y[i, i] += complex(self.Gs[i], self.Bs[i])

        # map from bus number to 0-based index
        busnum_to_idx = {int(self.bus_i[k]): k for k in range(nb)}

        for k in range(self.n_branch):
            fbus = int(self.branch[k, 0])
            tbus = int(self.branch[k, 1])
            r, x, b = self.branch[k, 2:5]
            ratio = self.branch[k, 8]
            angle_deg = self.branch[k, 9]
            status = int(self.branch[k, 10])

            if status == 0:
                continue

            # 索引
            f = busnum_to_idx[fbus]
            t = busnum_to_idx[tbus]

            z = complex(r, x)
            if abs(z) < 1e-12:
                # 避免除零（极少数数据 r=0 但 x>0，可直接用 1j/x）
                y = 1 / (1j * x) if abs(x) > 1e-12 else 1e12  # 非常大
            else:
                y = 1 / z
            jb = 1j * (b / 2.0)

            tap = ratio if ratio != 0 else 1.0
            shift_rad = math.radians(angle_deg)
            t_complex = tap * complex(math.cos(shift_rad), math.sin(shift_rad))  # t∠θ

            # 加入 π 模型
            # Y_ff, Y_tt, Y_ft, Y_tf
            Y[f, f] += (y + jb) / (t_complex * np.conj(t_complex))
            Y[t, t] += (y + jb)
            Y[f, t] += - y / np.conj(t_complex)
            Y[t, f] += - y / (t_complex)

        return Y

    def solve_feasibility(self, pA_pu, pB_pu):
        """
        给定两台机组(A,B)的有功出力（标幺），求解 AC 可行性（目标为常数0）。
        返回: (feasible(bool), solution(dict or None), time_sec(float))
        """
        t0 = time.time()
        try:
            m = pyo.ConcreteModel()

            m.BUS = pyo.RangeSet(1, self.n_bus)
            m.GEN = pyo.RangeSet(1, self.n_gen)

            # 变量
            m.Vm = pyo.Var(m.BUS, bounds=lambda _, i: (self.Vmin[i-1], self.Vmax[i-1]), initialize=1.0)
            m.Va = pyo.Var(m.BUS, bounds=(-math.pi, math.pi), initialize=0.0)  # 放宽相角范围
            m.Pg = pyo.Var(m.GEN, bounds=lambda _, g: (self.Pmin[g-1], self.Pmax[g-1]))
            m.Qg = pyo.Var(m.GEN, bounds=lambda _, g: (self.Qmin[g-1], self.Qmax[g-1]), initialize=0.0)

            # 固定两台扫描机组的 Pg
            def fix_pg_rule(_m):
                _m.Pg[self.genA_idx + 1].fix(pA_pu)
                _m.Pg[self.genB_idx + 1].fix(pB_pu)
            fix_pg_rule(m)

            # 参考母线相角
            slack = self.slack_bus
            m.ref = pyo.Constraint(expr=m.Va[slack] == 0.0)

            # 目标：0
            m.OBJ = pyo.Objective(expr=0.0, sense=pyo.minimize)

            G = self.G
            B = self.B

            # 功率平衡（注：允许一母线多台机组）
            busnum_to_idx = {int(self.bus_i[k]): k for k in range(self.n_bus)}

            def P_balance_rule(_m, i):
                # i 为 1-based 母线编号（按文件顺序）
                idx = i - 1
                Pi = 0.0
                for j in _m.BUS:
                    gij = G[idx, j-1]
                    bij = B[idx, j-1]
                    Pi += _m.Vm[i] * _m.Vm[j] * (gij * pyo.cos(_m.Va[i] - _m.Va[j]) + bij * pyo.sin(_m.Va[i] - _m.Va[j]))
                # 汇总各机组注入
                pg_sum = 0.0
                bus_no = int(self.bus_i[idx])
                for gi in self.bus_to_gens.get(bus_no, []):
                    pg_sum += _m.Pg[gi + 1]
                return pg_sum - self.Pd[idx] == Pi

            def Q_balance_rule(_m, i):
                idx = i - 1
                Qi = 0.0
                for j in _m.BUS:
                    gij = G[idx, j-1]
                    bij = B[idx, j-1]
                    Qi += _m.Vm[i] * _m.Vm[j] * (gij * pyo.sin(_m.Va[i] - _m.Va[j]) - bij * pyo.cos(_m.Va[i] - _m.Va[j]))
                qg_sum = 0.0
                bus_no = int(self.bus_i[idx])
                for gi in self.bus_to_gens.get(bus_no, []):
                    qg_sum += _m.Qg[gi + 1]
                return qg_sum - self.Qd[idx] == Qi

            m.Pbal = pyo.Constraint(m.BUS, rule=P_balance_rule)
            m.Qbal = pyo.Constraint(m.BUS, rule=Q_balance_rule)

            # 初值（可选更靠近额定）
            for i in m.BUS:
                m.Vm[i].value = max(min(self.Vm0[i-1], self.Vmax[i-1]), self.Vmin[i-1])
                m.Va[i].value = math.radians(self.Va0_deg[i-1])

            # 求解
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

            # 提取解（可选）
            sol = {
                'Pg_MW': [pyo.value(m.Pg[g]) * self.baseMVA for g in m.GEN],
                'Qg_MVar': [pyo.value(m.Qg[g]) * self.baseMVA for g in m.GEN],
                'Vm': [pyo.value(m.Vm[i]) for i in m.BUS],
                'Va_deg': [pyo.value(m.Va[i]) * 180.0 / math.pi for i in m.BUS]
            }
            return True, sol, t_used

        except Exception:
            # 任何异常返回不可行
            return False, None, time.time() - t0

    def run_scan(self):
        """
        进行二维网格扫描：Pg(genA) × Pg(genB)
        - 打印每个可行解的时间
        - 在达到100个可行解时计算并打印前100个的平均时间
        - 最后输出本次所有可行解的整体平均时间
        """
        print("\n开始扫描 ...")
        pA_vals_pu = np.linspace(self.pA_min_pu, self.pA_max_pu, self.n_points)
        pB_vals_pu = np.linspace(self.pB_min_pu, self.pB_max_pu, self.n_points)

        feasible_count = 0
        total_points = len(pA_vals_pu) * len(pB_vals_pu)
        processed = 0

        for pA in pA_vals_pu:
            for pB in pB_vals_pu:
                processed += 1
                ok, sol, tsec = self.solve_feasibility(pA, pB)
                if ok:
                    feasible_count += 1
                    self.feasible_times.append(tsec)
                    print(f"✅ 可行解 #{feasible_count:>4}: "
                          f"Pg(bus{self.genA_bus})={pA*self.baseMVA:.1f} MW, "
                          f"Pg(bus{self.genB_bus})={pB*self.baseMVA:.1f} MW, "
                          f"用时={tsec:.3f}s")
                    # 达到100个后，计算前100个的平均用时（仅打印一次）
                    if (not self.avg100_reported) and len(self.feasible_times) >= 100:
                        avg100 = sum(self.feasible_times[:100]) / 100.0
                        self.avg100_reported = True
                        print(f"🎯 前100个可行解平均求解时间：{avg100:.3f}s")
                # 进度提示（可选）
                if processed % max(1, total_points // 10) == 0 or processed == total_points:
                    prog = 100.0 * processed / total_points
                    print(f"进度: {processed}/{total_points} ({prog:.1f}%) | 已获可行解: {feasible_count}")

        if self.feasible_times:
            overall_avg = sum(self.feasible_times) / len(self.feasible_times)
            print(f"\n⏱️ 本次共获得 {len(self.feasible_times)} 个可行解；"
                  f"整体平均求解时间 = {overall_avg:.3f}s")
        else:
            print("\n未获得可行解。请适当放宽扫描范围或调整求解器设置。")

# ========== 运行示例 ==========
if __name__ == "__main__":
    # 可以根据需要调整扫描机组与网格密度
    # 默认选择 bus 65 与 66（两台较大机组），扫描区间为它们 Pmax 的 25%~75%
    analyzer = ACOPF118Analyzer(
        case_file='case118.m',
        genA_bus=65,
        genB_bus=66,
        n_points=15,  # 每个维度 15 点，共 225 次求解（可根据性能调整）
        solver_opts={
            'max_iter': 300,
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'print_level': 0
        }
    )
    analyzer.run_scan()