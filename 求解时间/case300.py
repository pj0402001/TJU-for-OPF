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
    要求文件中存在：
      mpc.baseMVA = <num>;
      mpc.bus = [ ... ];
      mpc.gen = [ ... ];
      mpc.branch = [ ... ];
    返回: dict(baseMVA, bus(np.ndarray), gen(np.ndarray), branch(np.ndarray))
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    def extract_block(name):
        pattern = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
        m = re.search(pattern, txt, flags=re.S | re.M)
        if not m:
            raise ValueError(f"无法在 {file_path} 中找到 {name} 数据块（请确认是 MATPOWER 案例文件）")
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
            nums = [float(tok) for tok in tokens if tok]
            if nums:
                rows.append(nums)
        return np.array(rows, dtype=float)

    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9.+\-Ee]+)\s*;", txt)
    if not m:
        raise ValueError("未找到 mpc.baseMVA（请确认提供的是 MATPOWER 案例文件）")
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

# ========== ACOPF 300-bus 分析器 ==========
class ACOPF300Analyzer:
    def __init__(self,
                 case_file='case300.m',
                 n_points=15,
                 genA_bus=None,
                 genB_bus=None,
                 range_frac=(0.35, 0.75),
                 per_bus_frac=None,
                 solver_opts=None):
        """
        case_file : MATPOWER 案例文件路径（例如 case300.m）
        n_points  : 每个维度扫描点数（总求解数为 n_points^2）
        genA_bus, genB_bus : 指定两台扫描机组所在母线号（不指定则自动选择 Pmax 最大的两台在运机组，尽量避开参考母线且不同母线）
        range_frac: 默认扫描相对范围（相对于 Pmax），当 per_bus_frac 未指定时使用
        per_bus_frac: 可按母线定制扫描范围的字典，例如 {191:(0.50,0.53), 119:(0.57,0.64)}
        solver_opts: Ipopt 求解器选项 dict
        """
        data = load_matpower_case(case_file)
        self.baseMVA = data['baseMVA']
        self.bus = data['bus']
        self.gen = data['gen']
        self.branch = data['branch']

        # 维度
        self.n_bus = self.bus.shape[0]
        self.n_gen = self.gen.shape[0]
        self.n_branch = self.branch.shape[0]

        # 母线数据（MATPOWER 列定义）
        # bus: [BUS_I, TYPE, PD, QD, GS, BS, AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN]
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

        # 发电机数据（MATPOWER 列定义）
        # gen: [GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, STATUS, PMAX, PMIN, ...]
        self.gen_bus = self.gen[:, 0].astype(int)
        self.gen_status = self.gen[:, 7].astype(int)
        self.Pmax = self.gen[:, 8] / self.baseMVA
        self.Pmin = self.gen[:, 9] / self.baseMVA
        self.Qmax = self.gen[:, 3] / self.baseMVA
        self.Qmin = self.gen[:, 4] / self.baseMVA
        self.Vg_set = self.gen[:, 5]

        # 仅考虑投运机组
        self.active_gen_idx = [i for i in range(self.n_gen) if self.gen_status[i] == 1]

        # 构建 母线号 -> 投运机组索引列表
        self.bus_to_gens = {int(b): [] for b in self.bus_i}
        for gi in self.active_gen_idx:
            self.bus_to_gens[int(self.gen_bus[gi])].append(gi)

        # 参考母线（type==3）
        slack_candidates = self.bus_i[self.bus_type == 3]
        if len(slack_candidates) == 0:
            raise ValueError("未找到参考母线 (bus type == 3)")
        self.slack_bus = int(slack_candidates[0])

        # 母线号 -> 行索引
        self.busnum_to_idx = {int(self.bus_i[k]): k for k in range(self.n_bus)}

        # 构建 Ybus（含变比/相移、π模型、母线并联导纳）
        self.Ybus = self.build_Ybus()
        self.G = self.Ybus.real
        self.B = self.Ybus.imag

        # 选择扫描机组
        if genA_bus is None or genB_bus is None:
            # 自动选择 Pmax 最大的两台在运机组，优先不同母线且避开参考母线
            cand = sorted(self.active_gen_idx, key=lambda i: self.Pmax[i], reverse=True)
            pick = []
            seen_bus = set()
            for gi in cand:
                b = int(self.gen_bus[gi])
                if b == self.slack_bus:
                    continue
                if b in seen_bus:
                    continue
                seen_bus.add(b)
                pick.append(gi)
                if len(pick) == 2:
                    break
            if len(pick) < 2:
                pick = cand[:2]
            self.genA_idx, self.genB_idx = pick[0], pick[1]
            self.genA_bus = int(self.gen_bus[self.genA_idx])
            self.genB_bus = int(self.gen_bus[self.genB_idx])
        else:
            self.genA_bus = int(genA_bus)
            self.genB_bus = int(genB_bus)
            self.genA_idx = self.find_active_gen_index_by_bus(self.genA_bus)
            self.genB_idx = self.find_active_gen_index_by_bus(self.genB_bus)

        # 扫描范围：支持 per_bus_frac 定制，否则使用 range_frac
        self.per_bus_frac = per_bus_frac or {}
        def frac_range_for(bus_no, gi):
            if bus_no in self.per_bus_frac:
                lo, hi = self.per_bus_frac[bus_no]
            else:
                lo, hi = range_frac
            # 夹紧到 [Pmin, Pmax]
            pmin = max(self.Pmin[gi], lo * self.Pmax[gi])
            pmax = min(self.Pmax[gi], hi * self.Pmax[gi])
            if pmin > pmax:
                # 极端情况下（例如 Pmin 高于指定上界），退化为 [Pmin, Pmin]
                pmin = pmax = self.Pmin[gi]
            return pmin, pmax

        self.pA_min_pu, self.pA_max_pu = frac_range_for(self.genA_bus, self.genA_idx)
        self.pB_min_pu, self.pB_max_pu = frac_range_for(self.genB_bus, self.genB_idx)

        self.n_points = int(n_points)
        self.solver_opts = solver_opts or {
            'max_iter': 400,
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'print_level': 0,
            'linear_solver': 'mumps'
        }

        # 统计
        self.feasible_times = []
        self.avg100_reported = False

        # 打印信息
        print("===== IEEE 300 节点 AC-OPF 可行性扫描 =====")
        print(f"BaseMVA: {self.baseMVA}")
        print(f"母线数: {self.n_bus}, 发电机数: {self.n_gen}, 支路数: {self.n_branch}")
        print(f"参考母线: {self.slack_bus}")
        total_Pd = np.sum(self.bus[:, 2])
        print(f"系统总有功负荷: {total_Pd:.1f} MW")
        print(f"扫描机组A: gen#{self.genA_idx+1} @ bus {self.genA_bus} (Pmax={self.Pmax[self.genA_idx]*self.baseMVA:.1f} MW)")
        print(f"扫描机组B: gen#{self.genB_idx+1} @ bus {self.genB_bus} (Pmax={self.Pmax[self.genB_idx]*self.baseMVA:.1f} MW)")
        print("扫描范围（MW）:")
        print(f"  A: [{self.pA_min_pu*self.baseMVA:.2f}, {self.pA_max_pu*self.baseMVA:.2f}]"
              f"  (占 Pmax 的 {100*self.pA_min_pu/self.Pmax[self.genA_idx]:.1f}% ~ {100*self.pA_max_pu/self.Pmax[self.genA_idx]:.1f}%)")
        print(f"  B: [{self.pB_min_pu*self.baseMVA:.2f}, {self.pB_max_pu*self.baseMVA:.2f}]"
              f"  (占 Pmax 的 {100*self.pB_min_pu/self.Pmax[self.genB_idx]:.1f}% ~ {100*self.pB_max_pu/self.Pmax[self.genB_idx]:.1f}%)")
        print(f"网格密度: {self.n_points} × {self.n_points} = {self.n_points*self.n_points} 点\n")

    def find_active_gen_index_by_bus(self, bus_no: int):
        idxs = [i for i in self.active_gen_idx if int(self.gen_bus[i]) == int(bus_no)]
        if not idxs:
            raise ValueError(f"未找到位于母线 {bus_no} 的在运发电机（或 STATUS=0）")
        return idxs[0]

    def build_Ybus(self):
        """
        构建带变比/相移的 Ybus；包含线路π模型与母线并联导纳 Gs+jBs。
        branch 列：
          fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
        ratio=0 表示无变比；angle 为度。
        """
        nb = self.n_bus
        Y = np.zeros((nb, nb), dtype=complex)

        # 母线并联导纳
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
            t_complex = tap * complex(math.cos(shift_rad), math.sin(shift_rad))  # t∠θ

            # π 模型组装
            Y[f, f] += (y + jb) / (t_complex * np.conj(t_complex))
            Y[t, t] += (y + jb)
            Y[f, t] += - y / np.conj(t_complex)
            Y[t, f] += - y / (t_complex)

        return Y

    def solve_feasibility(self, pA_pu, pB_pu):
        """
        给定两台机组(A,B)的有功出力（标幺），求解 AC 可行性（目标为常数 0）。
        返回: (feasible(bool), solution(dict or None), time_sec(float))
        """
        t0 = time.time()
        try:
            m = pyo.ConcreteModel()
            m.BUS = pyo.RangeSet(1, self.n_bus)
            m.GEN = pyo.RangeSet(1, self.n_gen)

            # Vm/Va
            m.Vm = pyo.Var(m.BUS, bounds=lambda _, i: (self.Vmin[i-1], self.Vmax[i-1]), initialize=1.0)
            m.Va = pyo.Var(m.BUS, bounds=(-math.pi, math.pi), initialize=0.0)

            # Pg/Qg 边界（停机机组上下界为(0,0)）
            def pg_bounds(_, g):
                gi = g - 1
                return (0.0, 0.0) if self.gen_status[gi] == 0 else (self.Pmin[gi], self.Pmax[gi])

            def qg_bounds(_, g):
                gi = g - 1
                return (0.0, 0.0) if self.gen_status[gi] == 0 else (self.Qmin[gi], self.Qmax[gi])

            m.Pg = pyo.Var(m.GEN, bounds=pg_bounds)
            m.Qg = pyo.Var(m.GEN, bounds=qg_bounds)

            # 显式处理停运机组 & 在运机组的初始化（夹紧到边界内）
            for g in m.GEN:
                gi = g - 1
                if self.gen_status[gi] == 0:
                    m.Pg[g].fix(0.0)
                    m.Qg[g].fix(0.0)
                else:
                    # 从 case 文件读取的初值（Pg: gen[:,1], Qg: gen[:,2]），转成 pu，并夹紧
                    pg0 = float(self.gen[gi, 1]) / self.baseMVA
                    qg0 = float(self.gen[gi, 2]) / self.baseMVA
                    pg0 = min(max(pg0, self.Pmin[gi]), self.Pmax[gi])
                    qg0 = min(max(qg0, self.Qmin[gi]), self.Qmax[gi])
                    m.Pg[g].value = pg0
                    m.Qg[g].value = qg0

            # 固定两台扫描机组的 Pg
            m.Pg[self.genA_idx + 1].fix(pA_pu)
            m.Pg[self.genB_idx + 1].fix(pB_pu)

            # 参考母线相角（注意：模型索引是按数组行序1..n，不是母线号）
            slack_pos = self.busnum_to_idx[self.slack_bus] + 1
            m.ref = pyo.Constraint(expr=m.Va[slack_pos] == 0.0)

            # 目标：0
            m.OBJ = pyo.Objective(expr=0.0, sense=pyo.minimize)

            G, B = self.G, self.B

            # 功率平衡
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

            # Vm/Va 初值
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

            # 安全取值函数
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
        二维网格扫描：Pg(genA_bus) × Pg(genB_bus)
        - 打印每个可行解的时间
        - 累计到100个可行解时，计算并打印前100个平均时间（仅一次）
        - 扫描结束打印整体平均时间
        """
        print("开始扫描 ...")
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
                          f"Pg(bus{self.genA_bus})={pA*self.baseMVA:.2f} MW, "
                          f"Pg(bus{self.genB_bus})={pB*self.baseMVA:.2f} MW, "
                          f"用时={tsec:.3f}s")
                    if (not self.avg100_reported) and len(self.feasible_times) >= 100:
                        avg100 = sum(self.feasible_times[:100]) / 100.0
                        self.avg100_reported = True
                        print(f"🎯 前100个可行解平均求解时间：{avg100:.3f}s")
                # 进度提示
                if processed % max(1, total_points // 10) == 0 or processed == total_points:
                    prog = 100.0 * processed / total_points
                    print(f"进度: {processed}/{total_points} ({prog:.1f}%) | 已获可行解: {feasible_count}")

        if self.feasible_times:
            overall_avg = sum(self.feasible_times) / len(self.feasible_times)
            print(f"\n⏱️ 本次共获得 {len(self.feasible_times)} 个可行解；整体平均求解时间 = {overall_avg:.3f}s")
        else:
            print("\n未获得可行解。可考虑：微调扫描范围、降低步数、或调整求解器参数。")

# ========== 运行示例 ==========
if __name__ == "__main__":
    # 按你的要求设置：bus191 在 50%~53% Pmax，bus119 在 57%~64% Pmax
    analyzer = ACOPF300Analyzer(
        case_file='case300.m',
        n_points=20,                 # 每维 15 点；区间较窄时可适当增大或减小
        genA_bus=191,                # 扫描机组A：母线 191
        genB_bus=119,                # 扫描机组B：母线 119
        per_bus_frac={191: (0.50, 0.53), 119: (0.57, 0.64)},
        solver_opts={
            'max_iter': 400,
            'tol': 1e-6,
            'acceptable_tol': 1e-4,
            'print_level': 0,
            'linear_solver': 'mumps'
        }
    )
    analyzer.run_scan()